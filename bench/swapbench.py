#!/usr/bin/env python3
"""Is swap on vroom better than swap on an SSD, for a program that has to swap?

Runs the same two workloads in a cgroup whose memory limit forces them to
swap, with only the place the swap goes changed from one run to the next:

  none    no limit at all: how fast the workload is when it does not swap
  ssd     a swap file on the filesystem holding --dir
  zswap   the same, with zswap in front of it
  zram    a zram device (zstd)
  vroom   a compressed vroom device
  vroom-plain, vroom-enc   a vroom device that does not compress; the second encrypts
  ssd-dev the whole disk or partition given as --ssd-dev, which is overwritten

The backend under test is given the highest swap priority, so what the
workload swaps out goes there; swap already in use is left alone. Each run
records wall time, how long the workload was stalled for want of memory
(the cgroup's memory.pressure), its CPU time, and how much it swapped.

Needs root. Results go to --out as JSON lines, and a table is printed.

Things it cannot make fair: zram keeps its pages in RAM that is not charged
to the cgroup, so here it gets that memory for nothing, which a machine
really short of memory would not; and the CPU time of the vroom daemon and
of the kernel's own reclaim threads is outside the cgroup's count.
"""
import argparse, json, os, random, shutil, statistics, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
CG = "/sys/fs/cgroup/swapbench"
PRIO = "32000"


def sh(*cmd, check=True, quiet=False):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and r.returncode:
        sys.exit(f"swapbench: {' '.join(cmd)} failed:\n{r.stdout}")
    if not quiet and r.stdout.strip() and os.environ.get("SWAPBENCH_VERBOSE"):
        print(r.stdout.strip())
    return r.stdout


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def swap_used_kb(dev):
    real = os.path.realpath(dev)
    for line in read("/proc/swaps").splitlines()[1:]:
        f = line.split()
        if os.path.realpath(f[0]) == real:
            return int(f[3])
    return 0


# ---- the backends: each sets up a swap device and returns (device, undo) ----

def be_none(a):
    return None, lambda: None


def ssd_on(a, disk=False):
    if disk:
        if not a.ssd_dev:
            sys.exit("swapbench: the ssd-dev backend needs --ssd-dev")
        dev = a.ssd_dev
    else:
        dev = os.path.join(a.dir, "swapbench.swap")
        sh("fallocate", "-l", f"{a.swap_gib}G", dev)
        os.chmod(dev, 0o600)
    sh("mkswap", "-f", dev, quiet=True)
    sh("swapon", "-p", PRIO, dev)

    def undo():
        sh("swapoff", dev)
        if disk:
            sh("wipefs", "-a", dev, quiet=True)
        else:
            os.unlink(dev)
    return dev, undo


def be_ssd(a):
    return ssd_on(a)


def be_ssd_dev(a):
    return ssd_on(a, disk=True)


def be_zswap(a):
    was = read("/sys/module/zswap/parameters/enabled").strip()
    write("/sys/module/zswap/parameters/enabled", "Y")
    dev, undo_ssd = ssd_on(a)

    def undo():
        undo_ssd()
        write("/sys/module/zswap/parameters/enabled", was or "N")
    return dev, undo


def be_zram(a):
    sh("modprobe", "zram")
    dev = sh("zramctl", "--find", "--size", f"{a.swap_gib}G", "--algorithm", "zstd").strip()
    sh("mkswap", dev, quiet=True)
    sh("swapon", "-p", PRIO, dev)

    def undo():
        sh("swapoff", dev)
        sh("zramctl", "--reset", dev)
    return dev, undo


def be_vroom(a, how=None):
    how = how or ["--compress", "--ratio", str(a.ratio)]
    sh("vroomctl", "add", "swapbench", "--size", str(a.vram_mib), "--prio", PRIO, *how, quiet=True)
    dev = [l.split("=", 1)[1] for l in read("/etc/vroom/swapbench.conf").splitlines()
           if l.startswith("VROOM_DEV=")][0]

    def undo():
        sh("vroomctl", "remove", "swapbench", quiet=True)
    return dev, undo


def be_vroom_plain(a):  # no compression, no single copies, no encryption: the card as a plain disk
    return be_vroom(a, [])


def be_vroom_enc(a):  # the same, with each block encrypted and checked
    return be_vroom(a, ["--encrypt"])


BACKENDS = {"none": be_none, "ssd": be_ssd, "ssd-dev": be_ssd_dev, "zswap": be_zswap, "zram": be_zram,
            "vroom": be_vroom, "vroom-plain": be_vroom_plain, "vroom-enc": be_vroom_enc}


# ---- the workloads: each returns (argv, memory limit in MiB) ----

def wl_touch(a):
    return ([sys.executable, os.path.join(HERE, "pagetouch.py"), str(a.touch_gib), str(a.touches)],
            int(a.touch_gib * 1024 * a.fraction))


def wl_compile(a):
    # Many compilations at once of a real, middling C file: the daemon's own source.
    src = os.path.join(HERE, "..", "vroom.c")
    job = (f"gcc -O2 -g -fsanitize=address,undefined -DFAKE_CARD -DJOB=$0 -c {src} -o /dev/null")
    script = f"seq {a.compile_jobs} | xargs -P {a.compile_par} -I@ sh -c '{job}' @"
    return (["sh", "-c", script], int(a.compile_par * 145 * a.fraction))


WORKLOADS = {"touch": wl_touch, "compile": wl_compile}


def pressure(text, kind):
    for line in text.splitlines():
        if line.startswith(kind):
            return int(line.split("total=")[1]) / 1e6
    return 0.0


def stat(text, key):
    for line in text.splitlines():
        f = line.split()
        if f and f[0] == key:
            return int(f[1])
    return 0


def run_one(a, backend, workload):
    argv, limit_mib = WORKLOADS[workload](a)
    dev, undo = BACKENDS[backend](a)
    try:
        sh("sync")
        write("/proc/sys/vm/drop_caches", "1")
        if os.path.isdir(CG):
            os.rmdir(CG)
        os.mkdir(CG)
        write(f"{CG}/memory.max", "max" if backend == "none" else f"{limit_mib}M")
        base = swap_used_kb(dev) if dev else 0
        peak = [0]
        done = threading.Event()

        def watch():
            while not done.wait(0.25):
                if dev:
                    peak[0] = max(peak[0], swap_used_kb(dev) - base)
        th = threading.Thread(target=watch)
        th.start()

        def enter():
            write(f"{CG}/cgroup.procs", str(os.getpid()))
        t = time.perf_counter()
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=enter)
        wall = time.perf_counter() - t
        done.set()
        th.join()
        r = {"backend": backend, "workload": workload, "wall_s": round(wall, 2), "rc": p.returncode,
             "limit_mib": 0 if backend == "none" else limit_mib}
        pr, ms, cs = read(f"{CG}/memory.pressure"), read(f"{CG}/memory.stat"), read(f"{CG}/cpu.stat")
        r["stall_some_s"] = round(pressure(pr, "some"), 2)
        r["stall_full_s"] = round(pressure(pr, "full"), 2)
        r["cpu_user_s"] = round(stat(cs, "user_usec") / 1e6, 2)
        r["cpu_sys_s"] = round(stat(cs, "system_usec") / 1e6, 2)
        r["swapped_out_mib"] = stat(ms, "pswpout") // 256
        r["swapped_in_mib"] = stat(ms, "pswpin") // 256
        r["zswapped_out_mib"] = stat(ms, "zswpout") // 256
        r["peak_mem_mib"] = int(read(f"{CG}/memory.peak", "0") or 0) >> 20
        r["peak_on_device_mib"] = peak[0] // 1024
        if p.returncode:
            r["error"] = (p.stderr or p.stdout)[-300:]
        else:
            try:
                r.update(json.loads(p.stdout.strip().splitlines()[-1]))
            except (ValueError, IndexError):
                pass
        return r
    finally:
        for _ in range(50):  # the cgroup can take a moment to empty
            try:
                os.rmdir(CG)
                break
            except OSError:
                time.sleep(0.1)
        undo()


def table(rows):
    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    def spread(xs):
        return (max(xs) - min(xs)) if len(xs) > 1 else 0.0
    for wl in sorted({r["workload"] for r in rows}):
        print(f"\n{wl}")
        extra = wl == "touch"
        head = f"  {'backend':11} {'runs':>4} {'wall s':>8} {'+/-':>6} {'stalled s':>9} {'sys s':>7} {'out MiB':>8} {'in MiB':>8}"
        if extra:
            head += f" {'touch p50':>10} {'p90':>8} {'p99':>8} {'fill s':>7} {'scan s':>7}"
        print(head)
        for be in BACKENDS:
            rs = [r for r in rows if r["workload"] == wl and r["backend"] == be and not r["rc"]]
            bad = len([r for r in rows if r["workload"] == wl and r["backend"] == be and r["rc"]])
            if not rs and not bad:
                continue
            if not rs:
                print(f"  {be:11} {0:4}   all {bad} runs failed")
                continue
            w = [r["wall_s"] for r in rs]
            line = (f"  {be:11} {len(rs):4} {med(w):8.1f} {spread(w):6.1f} {med([r['stall_some_s'] for r in rs]):9.1f}"
                    f" {med([r['cpu_sys_s'] for r in rs]):7.1f} {med([r['swapped_out_mib'] + r['zswapped_out_mib'] for r in rs]):8.0f}"
                    f" {med([r['swapped_in_mib'] for r in rs]):8.0f}")
            if extra:
                line += (f" {med([r['touch_p50_us'] for r in rs]):7.1f} us {med([r['touch_p90_us'] for r in rs]):8.1f}"
                         f" {med([r['touch_p99_us'] for r in rs]):8.1f} {med([r['fill_s'] for r in rs]):7.1f}"
                         f" {med([r['scan_s'] for r in rs]):7.1f}")
            if bad:
                line += f"   ({bad} failed)"
            print(line)
    print("\n  wall, stalled, sys: medians. +/-: slowest run less fastest. stalled: time some task in the")
    print("  workload was waiting for memory. out/in: what the workload swapped. touch: one random page.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="none,ssd,zswap,zram,vroom")
    ap.add_argument("--workloads", default="touch,compile")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--dir", default="/var/tmp", help="where the SSD swap file goes")
    ap.add_argument("--ssd-dev", help="the disk or partition for the ssd-dev backend: IT IS OVERWRITTEN")
    ap.add_argument("--swap-gib", type=int, default=8, help="size of the SSD swap file and the zram device")
    ap.add_argument("--vram-mib", type=int, default=6144)
    ap.add_argument("--ratio", default="1.5")
    ap.add_argument("--fraction", type=float, default=0.35, help="memory limit, as a part of what the workload wants")
    ap.add_argument("--touch-gib", type=float, default=4)
    ap.add_argument("--touches", type=int, default=200000)
    ap.add_argument("--compile-jobs", type=int, default=240)
    ap.add_argument("--compile-par", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="swapbench.jsonl")
    ap.add_argument("--report", action="store_true", help="only print the table for an existing --out")
    a = ap.parse_args()
    if a.report:
        table([json.loads(l) for l in open(a.out)])
        return
    if os.geteuid():
        sys.exit("swapbench: needs root (swapon, cgroups)")
    if "memory" not in read("/sys/fs/cgroup/cgroup.subtree_control"):
        sys.exit("swapbench: the memory controller is not enabled in /sys/fs/cgroup")
    backends = a.backends.split(",")
    workloads = a.workloads.split(",")
    for b in backends:
        if b not in BACKENDS:
            sys.exit(f"swapbench: no backend {b}")
    if any(b.startswith("vroom") for b in backends) and not shutil.which("vroomctl"):
        sys.exit("swapbench: vroomctl is not installed")
    rows = []
    rng = random.Random(7)
    for run in range(a.runs):
        order = [(b, w) for b in backends for w in workloads]
        rng.shuffle(order)  # so that whatever else the machine is doing falls on all alike
        for b, w in order:
            r = run_one(a, b, w)
            r["run"] = run
            rows.append(r)
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"run {run + 1}/{a.runs} {w:8} {b:11} {r['wall_s']:7.1f} s  stalled {r['stall_some_s']:6.1f} s"
                  f"  out {r['swapped_out_mib'] + r['zswapped_out_mib']} MiB"
                  + (f"  FAILED: {r.get('error', '')[-120:]}" if r["rc"] else ""), flush=True)
    table(rows)


if __name__ == "__main__":
    main()
