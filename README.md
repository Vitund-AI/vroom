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

## Requirements

- Linux with systemd and the `nbd` module, or `ublk_drv` (kernel 6.0+) for the ublk backend
- The proprietary NVIDIA driver, loaded for the card you want to use
- gcc and make to build

## Install

```bash
make deb
sudo apt install ./vroom_*.deb
```

Or without packaging: `sudo make install` (to `/usr/local`), then
`sudo systemctl daemon-reload`.

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
- `--boot` recreates the device at every boot. **Contents never survive a stop
  or reboot**: swap and filesystems are made fresh each time.

`vroomctl stop NAME` and `start NAME` release and reclaim the VRAM without
forgetting the definition, e.g. to lend the GPU to something else. `remove`
and `stop` refuse, and change nothing, if the device is busy. Each device is
the systemd unit `vroom@NAME` (logs: `journalctl -u vroom@NAME`), defined by
`/etc/vroom/NAME.conf`. `sudo vroomctl bench NAME` runs a read-only latency
test that is safe on a live device.

To move pages that are already in disk swap over to the VRAM, cycle the disk
swap: `swapoff /swap.img && swapon /swap.img`.

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

## License

MIT, see [LICENSE](LICENSE).
