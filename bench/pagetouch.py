#!/usr/bin/env python3
"""A workload that lives in swap: fill more memory than the cgroup allows,
then touch pages at random and time every touch. Nearly every touch is a
page fault served from swap, so the times are swap-in latency as a program
feels it. Prints one line of JSON.

usage: pagetouch.py GiB TOUCHES [NOISE_BYTES]
"""
import hashlib, json, mmap, random, sys, time

gib = float(sys.argv[1])
touches = int(sys.argv[2])
noise = int(sys.argv[3]) if len(sys.argv) > 3 else 1400  # incompressible bytes per page: about 2.5x in all
n = int(gib * 262144)


def page(i):
    if i % 10 == 0:  # a tenth of pages are copies of a few, as in real swap
        i %= 4096
    b = i.to_bytes(8, "little")
    return hashlib.shake_256(b).digest(noise) + b * ((4096 - noise) // 8)


buf = mmap.mmap(-1, n * 4096, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
t = time.perf_counter()
for i in range(n):
    buf[i * 4096:(i + 1) * 4096] = page(i)
fill = time.perf_counter() - t

rng = random.Random(1)
order = [rng.randrange(n) for _ in range(touches)]
lat = [0] * touches
clock = time.perf_counter_ns
t = time.perf_counter()
for k, i in enumerate(order):
    t0 = clock()
    buf[i * 4096]
    lat[k] = clock() - t0
rand = time.perf_counter() - t
lat.sort()

# and read the lot back in order, checking it, as a program resuming after a long sleep would
t = time.perf_counter()
bad = 0
for i in range(n):
    if buf[i * 4096:i * 4096 + 16] != page(i)[:16]:
        bad += 1
scan = time.perf_counter() - t


def pct(p):
    return lat[min(touches - 1, int(touches * p))] / 1000.0


print(json.dumps({
    "fill_s": round(fill, 2), "rand_s": round(rand, 2), "scan_s": round(scan, 2),
    "touch_p50_us": round(pct(0.50), 1), "touch_p90_us": round(pct(0.90), 1),
    "touch_p99_us": round(pct(0.99), 1), "touch_mean_us": round(sum(lat) / touches / 1000.0, 1),
    "bad": bad,
}))
sys.exit(1 if bad else 0)
