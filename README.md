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
than the VRAM behind it. A sample of real swap from a desktop compressed
2.2x. `vroomctl list` shows the ratio being achieved and how full the card is.

The catch is that the size is a promise made in advance. If the data does not
compress that well the card fills up and further writes are refused. For swap
that loses nothing (the page stays in RAM), but the kernel will keep trying,
so stay conservative: `--ratio` defaults to 1.5. It is not offered for
`--mount`, where a refused write would corrupt the filesystem.

`--spill PATH` removes the catch by adding a file on disk as a second tier.
When the card passes 90% full, the oldest pages move to the file; when it
falls below 70% they come back. With the file as large as the device, which
is the default, writes are never refused. Pages in the file are read at the
disk's speed, so this is for the cold tail, not a replacement for the card.

`vroomctl selftest --compress 1.5` on an RTX 4090, with test data that
compresses 2.4x:

| | Plain | Compressed |
|---|---|---|
| Sequential write | 4,100 MiB/s | 620 MiB/s |
| Sequential read | 3,550 MiB/s | 1,170 MiB/s |
| 4K random write | 9.5 µs | 14.5 µs |
| 4K random read | 9.0 µs | 10.9 µs |

With a quarter of the pages on an NVMe SSD, random reads averaged 30 µs.

Setting `VROOM_COUNT=1` in a device's `/etc/vroom/NAME.conf` makes the daemon
count, in `/run/vroom/DEVICE.count`, how many pages written were already held
somewhere on the device. It changes nothing else; it is there to find out
whether keeping a single copy would be worth building.

Limits to know about: it costs 8 bytes of RAM per page offered (2 MiB per
GiB); the two water marks are fixed, not learned; and "oldest" is by when a
page was last written or last moved. It has had the self-test and one live
run as swap (3 GiB of incompressible pages pushed out of a memory-limited
process, a third of them onto disk, all read back intact), not months of
daily use: treat it as experimental.

`vroomctl stop NAME` and `start NAME` release and reclaim the VRAM without
forgetting the definition, e.g. to lend the GPU to something else. `remove`
and `stop` refuse, and change nothing, if the device is busy. Each device is
the systemd unit `vroom@NAME` (logs: `journalctl -u vroom@NAME`), defined by
`/etc/vroom/NAME.conf`. `sudo vroomctl bench NAME` runs a read-only latency
test that is safe on a live device. `vroomctl stats [NAME]` shows what a device
has read and written and, where they are switched on, its compression and
repeat counts; `watch -n 2 vroomctl stats` keeps it on screen.

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
  or the GPU goes away while a device is in use (driver upgrade,
  `rmmod nvidia`, GPU reset, passing the card to a VM), every process with
  swapped pages there takes I/O errors and dies, and mounted files are lost.
  Always `vroomctl stop` or `remove` first. The daemon ignores SIGTERM while
  its device is swap or mounted and the service never sends SIGKILL, but
  nothing can stop a manual `kill -9`.
- **Use `vroomctl stop`, not `systemctl stop`, on a busy device.** vroomctl
  fails at once and leaves everything running; systemctl waits up to 15
  minutes for the device to be freed.
- **Other GPU users get what is left.** The driver never evicts vroom's
  memory; other applications simply fail to allocate. Use `--size` to
  leave them room.
- **Mounting over a directory in use** (such as a live `/tmp`) hides what is
  already there, including sockets that running programs depend on.
- **Suspend** only works if the driver saves VRAM (`vroomctl list` warns if
  `PreserveVideoMemoryAllocations` is off).
- **Swapping to a userspace daemon can deadlock under extreme memory
  pressure.** Mitigations in place: `mlockall`, page-locked staging buffers,
  `PR_SET_IO_FLUSHER`, OOM score −1000. Keep disk swap enabled as the
  lower-priority fallback.
- Removing a swap device needs somewhere to put its pages. If RAM and other
  swap are full, swapoff fails and the device stays active.

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

## Roadmap

Ideas, not promises:

- **Encrypt what is stored**, on the card and in the spill file, with a key
  that exists only in the daemon's memory, so that nothing readable is left
  behind in a spill file, a suspend image or a card handed to a VM.
- **Keep one copy of identical pages.** `VROOM_COUNT=1` exists to find out
  how often that would help.
- **Learn the water marks** for spilling instead of fixing them at 90% and 70%.
- **Give VRAM back when something else wants it.** Watch how much of the
  card other programs are using and, between a high and a low mark, move
  pages to the spill file and release whole 256 MiB allocations, taking
  them back later. It can only keep room free in advance: an allocation by
  another program fails at once, with no chance to make room for it.
- **Do not hold up other reads** while one waits for the spill file.
- **GPUs other than NVIDIA's**, through Vulkan or OpenCL.

## License

MIT, see [LICENSE](LICENSE).
