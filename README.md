# vroom — idle NVIDIA VRAM as swap or scratch disk

Got a GPU sitting in a Linux box doing nothing? vroom turns its video memory
into fast block devices: swap, a tmpfs-style scratch filesystem, or both.

A small daemon allocates VRAM through the NVIDIA driver's own CUDA library
(`libcuda.so.1`, shipped with the driver; no CUDA toolkit needed) and serves it
to the kernel as a ublk or NBD block device. Each page in or out is one
host↔GPU copy over PCIe. There is no kernel module, so it works with Secure
Boot.

Measured on an RTX 3070 (4K random reads against the live block device):

| Backend | One reader | Under parallel load |
|---|---|---|
| ublk (default) | 7.0 µs | 210k IOPS |
| NBD | 9.3 µs | 197k IOPS |

Sequential throughput is 3.5–4 GiB/s. That is good-NVMe class, not RAM class:
about 4 µs of every page is the NVIDIA driver performing the copy.

`vroomctl selftest --size 2048` on two cards, which anyone can repeat without
root. It exercises the daemon's NBD path without a kernel block device, so
its figures sit a little below the live ublk device above.

| | RTX 3070 | RTX 4090 |
|---|---|---|
| VRAM | 8 GiB | 24 GiB |
| Host CPU | Ryzen 9 9900X | Ryzen 9 9900X |
| Host RAM | 32 GiB | 32 GiB |
| GPU link | PCIe 4.0 x16 | PCIe 4.0 x16 |
| Sequential write | 3,951 MiB/s | 4,149 MiB/s |
| Sequential read | 3,685 MiB/s | 3,550 MiB/s |
| 4K random write, 1 thread | 115k IOPS (8.7 µs) | 105k IOPS (9.5 µs) |
| 4K random read, 1 thread | 115k IOPS (8.7 µs) | 111k IOPS (9.0 µs) |
| 4K random read, 4 threads | 193k IOPS | 171k IOPS |

Both hosts ran Ubuntu 26.04, kernel 7.0 and NVIDIA driver 595. The 4090
figures are the median of three runs, the 3070's a single run. The faster
card is no faster: the time goes to the driver and the PCIe copy, not the
GPU, so what a bigger card buys is capacity.

## Requirements

- Linux with systemd and the `nbd` module, or `ublk_drv` (kernel 6.0+) for the ublk backend
- NVIDIA's own driver (with either its proprietary or its open kernel
  modules; not nouveau), loaded for the card you want to use
- gcc and make to build

Tested on Ubuntu 26.04 with kernel 7.0 and NVIDIA driver 595 (open kernel
modules): in daily use as swap on an RTX 3070, and the self-test on an RTX 4090.

## Install

```bash
git clone https://github.com/Vitund-AI/vroom.git
```

```bash
cd vroom
```

On Debian and Ubuntu, build a package and install it:

```bash
make deb
```

```bash
sudo apt install ./vroom_*.deb
```

On any other distribution, install straight to `/usr/local`:

```bash
sudo make install
```

```bash
sudo systemctl daemon-reload
```

Run `vroomctl selftest` before adding a device: it needs no root, changes
nothing, and tells you at once whether the driver and card are usable.

### Upgrade

Install the new package, or run `sudo make install` again. Devices that are
running keep the old daemon until they are next stopped and started, so
nothing in use is disturbed. The package tells Ubuntu's `needrestart` to
leave vroom's daemons alone; before 0.3.0 it would drain and restart them
during an upgrade.

### Uninstall

```bash
sudo apt remove vroom
```

This stops every device first, and refuses to go on if one is busy, so that
nothing holding live data is cut off. `apt purge` also deletes the device
definitions in `/etc/vroom`. After a `make install`, remove each device with
`sudo vroomctl remove NAME`, then:

```bash
sudo make uninstall
```

## Use

```bash
vroomctl selftest                                   # no root: verify + benchmark 1 GiB of VRAM
sudo vroomctl add swap --swap                       # all free VRAM as high-priority swap
vroomctl list
sudo vroomctl remove swap                           # swapoff, then release the VRAM
```

One card can be split into several devices, each with its own job:

```bash
sudo vroomctl add swap    --size 5000 --swap --boot
sudo vroomctl add scratch --size 2048 --mount /mnt/scratch
sudo vroomctl add blk     --size 512  --raw           # appears as /dev/vroom/blk
sudo vroomctl add safe    --size 2048 --mount /mnt/safe --encrypt   # encrypted, and checked for bit rot
```

```
NAME         STATE     DEVICE             SIZE  USE
blk          active    /dev/ublkb2     512 MiB  raw, /dev/vroom/blk
scratch      active    /dev/ublkb1    2048 MiB  mounted at /mnt/scratch, 287 MiB used
swap         active    /dev/ublkb0    5000 MiB  swap, priority 100, 0 MiB used
```

- `--swap` (default) runs `mkswap` and `swapon` at priority 100, so the VRAM
  fills before disk swap. `--prio N` changes that.
- `--mount DIR` formats the device as ext4 (no journal) and mounts it with
  mode 1777, like tmpfs. Unlike tmpfs, cached file data can be dropped from
  RAM once written to the card. `--mode` changes the directory mode.
- `--raw` only creates the block device.
- `--size MiB` defaults to all free VRAM less 256 MiB.
- `--backend ublk|nbd` picks the kernel interface; ublk is used when the
  kernel has it.
- `--gpu N` picks the card on a machine with several (default 0). N is
  CUDA's index, which by default puts the fastest card first and so can
  differ from the order `nvidia-smi` and `vroomctl list` show.
  `vroomctl selftest --gpu N` names the card it used.
- `--boot` recreates the device at every boot. **Contents never survive a stop
  or reboot**: swap and filesystems are made fresh each time.

## Compression and spilling to disk (experimental)

```bash
sudo vroomctl add swap --compress                        # device is 1.5x the VRAM
sudo vroomctl add swap --spill /var/lib --ratio 2.2      # 2.2x, overflowing to a file
```

`--compress` runs every page through zstd on its way to the card (using
`libzstd.so.1`, which systemd already depends on) and offers a device larger
than the VRAM behind it. Pages with identical contents are stored once: each
page gets a keyed 128-bit fingerprint, and one whose fingerprint is already
held just points at that copy, whether it is on the card or in the spill
file.
What is stored is encrypted with AES-256-GCM (from `libcrypto.so.3`) under a
key made up when the device starts and kept only in the daemon's memory, so
nothing readable is left on the card or in a spill file afterwards, and a
stored page that has been altered, or put back as it was at an earlier
writing, is refused instead of returned.
`--cipher chacha20` uses ChaCha20-Poly1305, for a CPU without AES
instructions.

A write does not wait to be compressed. The page is encrypted as it is, put
on the card and the write answered; one background thread compresses such
pages afterwards, once free room on the card runs below an eighth or the
card is half full, and a page that has been overwritten or discarded by then
is never compressed at all. Only when writes arrive faster than that thread
can win room back are they compressed on arrival, as they would otherwise
all have to be. `vroomctl stats` says how much is waiting and what share of
writes went which way. A device with `--spill` still compresses every page on
arrival.

A
sample of real swap from a host running several VMs compressed 2.2x, and one
page in eight written there was a copy of another. `vroomctl list` shows the
ratio being achieved and how full the card is.

The catch is that the size is a promise made in advance. If the data does not
compress that well the card fills up and further writes are refused. For swap
that loses nothing (the page stays in RAM), but the kernel will keep trying,
so stay conservative: `--ratio` defaults to 1.5. It is not offered for
`--mount`, where a refused write would corrupt the filesystem.

`--spill PATH` removes the catch by adding a file on disk as a second tier.
PATH can also be a whole empty disk or partition, such as `/dev/nvme1n1`,
which is then used directly, with no filesystem in between; it has to look
empty (no partitions, no filesystem) and not be in use, and everything on
it is overwritten. vroom labels a disk it uses and checks before every
start that the disk either carries the label or is still empty, so a name
like `/dev/sdb` that has come to mean a different disk is refused.
A spill file is made by the daemon, must not exist beforehand, and has no
name while it is in use (`df` counts it, `ls` does not show it), so it is
gone when the daemon is, however that happens. It needs a filesystem that
can do direct I/O, which rules out tmpfs.
When the card passes 90% full, pages move to the file: the oldest first,
but not those that have been read since the last time round, so what is
read again and again stays on the card and what is written and left alone
goes to the disk. When the card falls below 70% they come back, starting
with whatever was read from the file most recently. With the file as large as the device, which
is the default, writes are not refused for want of room. If the disk itself
fails, pages already on it are lost with it, and writes are refused once
the card is full instead of being left to wait. Pages in the file are read
at the disk's speed, so this is for the cold tail, not a replacement for
the card.

`vroomctl selftest --compress 1.5` on an RTX 4090, with test data that
compresses 2.4x and in which one page in eight is a copy of another, so
that it takes 2.8x less room:

| | Plain | Compressed |
|---|---|---|
| Sequential write | 4,100 MiB/s | 660 MiB/s |
| Sequential read | 3,550 MiB/s | 880 MiB/s |
| 4K random write | 9.5 µs | 13.5 µs |
| 4K random read | 9.0 µs | 11.0 µs |

With six in ten pages on an NVMe SSD, random reads averaged 68 µs for one
reader, and four readers together got 3.6 times as many done: a read waiting
for the disk does not hold up the others.

Setting `VROOM_COUNT=1` in a device's `/etc/vroom/NAME.conf` makes the daemon
count, in `/run/vroom/DEVICE.count`, how many pages written were already held
somewhere on the device. It changes nothing else; it is there to find out
whether keeping a single copy would be worth building.

Limits to know about: it costs about 50 bytes of RAM per page offered
(12 MiB per GiB), locked in memory, and 32 bytes of VRAM per page stored;
plain devices are encrypted only if asked (`--encrypt`); two pages are taken to be identical on
the strength of their fingerprints, without comparing them (for any two
pages the chance of a false match is below 1 in 2^95, far smaller than
that of the hardware getting it wrong); the two water marks are fixed, not
learned; "oldest" is by when a page was last written or last moved, and
"read since the last time round" is all that is known about reads: nothing
is predicted, and a page is spared for one turn through the card, not for
longer.
Storing one copy has the side channel it has everywhere (KSM, deduplicating
filesystems): a local user who can time their own swap-outs may be able to
tell that a page they wrote was already held for someone else. The counts
that would show it directly are readable by root only. It has had the self-test and a
few live runs as swap (gigabytes of pages pushed out of a memory-limited
process, part of them onto disk, all read back intact), not months of daily
use: treat it as experimental.

`vroomctl stop NAME` and `start NAME` release and reclaim the VRAM without
forgetting the definition, e.g. to lend the GPU to something else. `remove`
and `stop` refuse, and change nothing, if the device is busy. Each device is
the systemd unit `vroom@NAME` (logs: `journalctl -u vroom@NAME`), defined by
`/etc/vroom/NAME.conf`. `sudo vroomctl bench NAME` runs a read-only latency
test that is safe on a live device. `vroomctl stats [NAME]` shows what a device
holds and a table of what it has done: totals and the busiest two seconds
since it started, and for the last minute and the last five the rate and the
median, 95th and 99th percentile time, each for whole requests, for copies
to and from the card and for the spill file. The daemon times one operation
in sixteen. Compression, sharing, encryption and repeat counts are shown
where they apply; `watch -n 2 vroomctl stats` keeps it on screen.

To move pages that are already in disk swap over to the VRAM, cycle the disk
swap: `swapoff /swap.img && swapon /swap.img`.

## If it does not work

- **`vroomctl selftest` fails:** the NVIDIA driver is not loaded for that
  card, or `libcuda.so.1` is missing. `nvidia-smi` should list the card; if
  it does not, fix the driver first.
- **`could not start NAME; nothing was added`:** the daemon's reason is in
  `journalctl -u vroom@NAME`. Most often the card does not have that much
  free VRAM, so try a smaller `--size`.
- **No ublk:** kernels before 6.0, and some built without `ublk_drv`, only
  have NBD. vroom falls back to it by itself; `--backend nbd` forces it.
- **`NAME is still in use; nothing was stopped`:** something has files open
  on the mount, or there is not enough RAM and other swap to take the pages
  back. Free one or the other and try again.

## Things that will hurt

- **The VRAM is the only copy of the data.** If the daemon, the NVIDIA driver
  or the GPU goes away while a device is in use (`rmmod nvidia`, NVIDIA's
  `.run` installer, a GPU reset, passing the card to a VM), every process with
  swapped pages there takes I/O errors and dies, and mounted files are lost.
  Always `vroomctl stop` or `remove` first. The daemon ignores every signal
  it can while its device is swap, mounted or held by something on top of
  it (LVM, dm-crypt, a partition), and the service never sends SIGKILL, but
  nothing can stop a manual `kill -9`.
- **The card stays awake.** A program holding VRAM through CUDA keeps the
  GPU out of its lowest power states for as long as it runs. On a desktop
  that is a few watts; on a laptop with a GPU that would otherwise power
  down it costs battery, and vroom is probably the wrong tool there.
- **Consumer cards have no ECC.** A compressed device makes up for some of
  that. A stored page that fails its check is read again, and if it is
  still wrong, tried with each of its bits turned over until the check
  passes: one flipped bit, on the card or in the spill, is found and put
  right where it lies, in a few tens of milliseconds. More damage than that
  is an I/O error, never wrong data. `vroomctl stats` counts both. None of
  this helps a page that was already wrong in the machine's own memory
  before it was written. A device that is not compressed gets the same
  check and repair with `--encrypt`; without it there is no check at all,
  and it returns whatever the card holds, as a RAM disk on non-ECC memory
  would.
- **A plain device holds pages as they are** unless it was added with
  `--encrypt`. Anyone who can read the card's memory can read them, and
  they are wiped when the device is stopped cleanly. `--encrypt` stores
  every 4K block encrypted under a key that exists only in the daemon's
  memory, with 32 bytes in front of it to check it by: the device is 1/129th
  smaller than the VRAM it takes, and a little slower. It works for mounted
  and raw devices as well as swap.
- **After a driver upgrade, leave devices running until you reboot.** Ubuntu's
  driver packages replace the libraries but leave the old kernel module
  loaded, so a running device carries on unharmed. A device started after
  the upgrade would get the new libraries, which the old module refuses, so
  nothing can be started until reboot. `vroomctl list` warns when this is
  the case, and `stop` and `remove` then ask for `--force`.
- **Use `vroomctl stop`, not `systemctl stop`, on a busy device.** vroomctl
  fails at once and leaves everything running; systemctl waits up to 15
  minutes for the device to be freed.
- **Other GPU users get what is left.** The driver never evicts vroom's
  memory; other applications simply fail to allocate. Use `--size` to
  leave them room.
- **Mounting over a directory in use** (such as a live `/tmp`) hides what is
  already there, including sockets that running programs depend on.
- **Suspend and hibernate: swap is taken away first, and put back after.**
  Once the machine starts to sleep the daemon is frozen, and anything that
  then needs to swap would wait on it for ever. So `vroom-sleep.service`
  (enabled when the first swap device is added) empties and stops every
  swap device before the machine sleeps and before the NVIDIA driver saves
  the card, and starts them again, empty, on waking. Emptying needs room
  in RAM and other swap for what the device holds; if there is none, the
  machine does not go to sleep, and the journal says why. This has been
  rehearsed by starting the unit by hand, not yet through a real suspend:
  treat it as new. Mounted and raw devices cannot be emptied anywhere and
  are left alone; they survive only if the driver saves VRAM (`vroomctl
  list` warns if `PreserveVideoMemoryAllocations` is off).
- **Swapping to a userspace daemon can deadlock under extreme memory
  pressure.** Mitigations in place: `mlockall`, page-locked staging buffers,
  `PR_SET_IO_FLUSHER`, OOM score −1000. Keep disk swap enabled as the
  lower-priority fallback.
- Removing a swap device needs somewhere to put its pages. If RAM and other
  swap are full, swapoff fails and the device stays active.

## Testing

`make check` builds the daemon against a card made of ordinary memory and
runs its self-tests under the address and undefined-behaviour sanitizers.
It needs no GPU and no root, and is what CI runs. It covers the storage
engine: integrity through rewrites, cleaning, spilling and coming back,
several writers at once, discards, a damaged record, an altered page, a
flipped bit on the card and on the disk, a
failing spill disk, and a check after each stage that the store's tables
agree with each other. `vroomctl selftest` runs the same tests on a real
card. Neither covers the kernel side (the ublk queue, attaching, signals,
`vroomctl` itself), which has only been tested by hand.

## How it works

```
kernel swap ──► /dev/nbd0   ──► socketpair ──┐
            └─► /dev/ublkb0 ──► io_uring  ───┴► vroom worker thread
                                                 │ cuMemcpyHtoD / cuMemcpyDtoH
                                                 ▼
                                      VRAM, allocated in 256 MiB chunks
```

- **NBD:** the daemon attaches the kernel's NBD client to in-process
  socketpairs (one per worker thread) and speaks the NBD transmission protocol
  itself, so there is no network, no `nbd-client` and no separate server.
- **ublk:** requests arrive as io_uring commands on `/dev/ublkcN` (raw
  syscalls, no liburing). There is no socket and one copy fewer. CUDA
  serialises copies within a context, so one queue is fastest; several queues
  were measurably slower.

## Related projects

- [vramfs](https://github.com/Overv/vramfs): a FUSE filesystem in VRAM
  through OpenCL, so not tied to NVIDIA. The original, and by its own
  README not meant for serious use.
- [nbd-vram](https://github.com/c0deJedi/nbd-vram): VRAM as swap through the
  CUDA driver API and NBD, the closest relative. It was the first to take the swap away
  before suspend and put it back afterwards.
- [nbdkit-vram-plugin](https://libguestfs.org/nbdkit-vram-plugin.1.html):
  an nbdkit plugin over OpenCL, for any GPU with OpenCL.

If all you want is more swap and the RAM is not already full of things that
do not compress, zram costs nothing and needs no GPU. vroom is for memory
that would otherwise sit idle on a card, and the two stack: zram first,
vroom next, disk last.

## Roadmap

Ideas, not promises, in the order they are likely to be done:

1. **Soak the compressed store** on a real workload, and a stress test that
   drives a machine deep into swap to see that the precautions against
   deadlock hold.
2. **Give VRAM back when something else wants it.** Watch how much of the
   card other programs are using and, between a high and a low mark, move
   pages to the spill file and release whole 256 MiB allocations, taking
   them back later. It can only keep room free in advance: an allocation by
   another program fails at once, with no chance to make room for it.
3. **Learn the water marks** for spilling instead of fixing them at 90% and
   70%, if real use shows the fixed ones to be wrong.
4. **GPUs other than NVIDIA's**, through Vulkan or OpenCL.
5. A `--help` for `vroomctl`, packages other than `.deb`.

## License

MIT, see [LICENSE](LICENSE).
