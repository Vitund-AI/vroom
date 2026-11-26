/*
 * SPDX-License-Identifier: MIT
 * vroom - expose idle NVIDIA VRAM as a block device (for swap).
 *
 * VRAM is allocated through the driver's own CUDA library (libcuda.so.1,
 * shipped with the NVIDIA driver; no CUDA toolkit needed) and served to the
 * kernel as /dev/nbdN over in-process socketpairs speaking the NBD
 * transmission protocol. Every block I/O is one host<->device copy across PCIe.
 *
 *   vroom --selftest [--size MiB]          integrity + speed test, no root
 *   vroom --dev /dev/ublkb0 [--size MiB]   serve the device (root)
 *
 * With --compress RATIO pages are compressed on their way in and the device is
 * that much larger than the VRAM; --spill FILE lets it overflow to a file.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/io_uring.h>
#include <linux/nbd.h>
#include <linux/ublk_cmd.h>
#include <limits.h>
#include <stddef.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef PR_SET_IO_FLUSHER
#define PR_SET_IO_FLUSHER 57
#endif
#ifndef NBD_FLAG_CAN_MULTI_CONN
#define NBD_FLAG_CAN_MULTI_CONN (1 << 8)
#endif

#define MiB (1024ULL * 1024)
#define CHUNK (256 * MiB) /* VRAM is allocated in pieces of this size */
#define BUFSZ (1 * MiB)   /* per-thread pinned staging buffer */
#define BLK 4096
#define MAX_THREADS 16

/* ---- minimal CUDA driver API, resolved at runtime ---- */
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
static struct {
	int (*Init)(unsigned);
	int (*DeviceGet)(int *, int);
	int (*DeviceGetName)(char *, int, int);
	int (*CtxCreate)(CUcontext *, unsigned, int);
	int (*CtxSetCurrent)(CUcontext);
	int (*MemGetInfo)(size_t *, size_t *);
	int (*MemAlloc)(CUdeviceptr *, size_t);
	int (*MemAllocHost)(void **, size_t);
	int (*MemsetD8)(CUdeviceptr, unsigned char, size_t);
	int (*HtoD)(CUdeviceptr, const void *, size_t);
	int (*DtoH)(void *, CUdeviceptr, size_t);
} cu;

static CUcontext ctx;
static void cs_init(void);
static double cratio;   /* --compress: how much larger than the VRAM the device is; 0: off */
static uint64_t phys_size; /* VRAM behind the device; the same as dev_size unless compressing */
static CUdeviceptr *chunks;
static uint64_t dev_size;

static void die(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "vroom: ");
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	va_end(ap);
	exit(1);
}

static void *sym(void *lib, const char *name)
{
	void *p = dlsym(lib, name);
	if (!p)
		die("libcuda.so.1 lacks %s", name);
	return p;
}

static void vram_init(int gpu, uint64_t size_mib)
{
	void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
	if (!lib)
		die("cannot load libcuda.so.1 (is the NVIDIA driver installed?): %s", dlerror());
	cu.Init = sym(lib, "cuInit");
	cu.DeviceGet = sym(lib, "cuDeviceGet");
	cu.DeviceGetName = sym(lib, "cuDeviceGetName");
	cu.CtxCreate = sym(lib, "cuCtxCreate_v2");
	cu.CtxSetCurrent = sym(lib, "cuCtxSetCurrent");
	cu.MemGetInfo = sym(lib, "cuMemGetInfo_v2");
	cu.MemAlloc = sym(lib, "cuMemAlloc_v2");
	cu.MemAllocHost = sym(lib, "cuMemAllocHost_v2");
	cu.MemsetD8 = sym(lib, "cuMemsetD8_v2");
	cu.HtoD = sym(lib, "cuMemcpyHtoD_v2");
	cu.DtoH = sym(lib, "cuMemcpyDtoH_v2");

	int dev, r;
	char name[128] = "?";
	size_t free_b, total_b;
	if ((r = cu.Init(0)))
		die("cuInit failed (CUDA error %d)", r);
	if ((r = cu.DeviceGet(&dev, gpu)))
		die("no CUDA GPU with index %d (error %d)", gpu, r);
	cu.DeviceGetName(name, sizeof name, dev);
	if ((r = cu.CtxCreate(&ctx, 0, dev)))
		die("cuCtxCreate failed (error %d)", r);
	if ((r = cu.MemGetInfo(&free_b, &total_b)))
		die("cuMemGetInfo failed (error %d)", r);

	if (!size_mib) { /* default: everything free, less a little headroom */
		if (free_b < 768 * MiB)
			die("only %llu MiB of VRAM free", free_b / MiB);
		size_mib = free_b / MiB - 256;
	}
	phys_size = dev_size = size_mib * MiB;
	fprintf(stderr, "vroom: %s: %llu MiB free of %llu MiB, taking %llu MiB\n", name,
		free_b / MiB, total_b / MiB, (unsigned long long)size_mib);
	if (cratio) {
		if (size_mib < 16)
			die("--compress needs at least 16 MiB");
		dev_size = (uint64_t)(size_mib * cratio) * MiB;
		fprintf(stderr, "vroom: compressing: offering %llu MiB (%.2fx)\n",
			(unsigned long long)(dev_size / MiB), cratio);
	}

	uint64_t n = (phys_size + CHUNK - 1) / CHUNK;
	chunks = calloc(n, sizeof *chunks);
	for (uint64_t i = 0; i < n; i++) {
		uint64_t len = phys_size - i * CHUNK < CHUNK ? phys_size - i * CHUNK : CHUNK;
		if ((r = cu.MemAlloc(&chunks[i], len)))
			die("VRAM allocation failed at %llu MiB (CUDA error %d)",
			    (unsigned long long)(i * CHUNK / MiB), r);
		if ((r = cu.MemsetD8(chunks[i], 0, len)))
			die("VRAM clear failed (CUDA error %d)", r);
	}
	if (cratio)
		cs_init();
}

static int vram_rw(int wr, uint64_t off, char *buf, size_t len)
{
	while (len) {
		uint64_t ci = off / CHUNK, co = off % CHUNK;
		size_t n = len < CHUNK - co ? len : CHUNK - co;
		int r = wr ? cu.HtoD(chunks[ci] + co, buf, n) : cu.DtoH(buf, chunks[ci] + co, n);
		if (r) {
			fprintf(stderr, "vroom: VRAM %s failed at offset %llu (CUDA error %d)\n",
				wr ? "write" : "read", (unsigned long long)off, r);
			return -1;
		}
		off += n, buf += n, len -= n;
	}
	return 0;
}

/*
 * ---- compressed store (--compress RATIO) ----
 *
 * Each 4K page is compressed by itself and appended, behind an 8-byte header
 * naming it, to a log laid over the VRAM in 1 MiB segments. A table in RAM
 * says where the current copy of every page is. Overwriting or discarding a
 * page only marks the old copy dead; a segment with nothing live in it is
 * free again, and when free segments run short the one with the least live
 * data is read back and its survivors appended afresh.
 *
 * Writing a request is one copy to the card however many pages it holds, and
 * pages written together are read back together, as swap tends to do.
 *
 * The device is advertised RATIO times larger than the VRAM behind it. If the
 * data does not compress that well the store fills up and writes fail with
 * ENOSPC. For swap that is survivable (the page stays in RAM); a filesystem
 * would be corrupted, so vroomctl only offers this for swap.
 *
 * With --spill FILE there is a second, slower tier, so that a full card
 * overflows instead of refusing. The file is cut into the same 1 MiB pieces.
 * Above the high-water mark each write also moves about 1 MiB of the oldest
 * segments' live records to the file; below the low-water mark a thread
 * brings the most recently moved ones back. Pages in the file are read
 * straight from it.
 */
#define SEG (1 * MiB)
#define REC_HDR 8
#define RAW_ABOVE 3584 /* a page that does not get below this is stored as it is */
#define STAGE_BYTES (BUFSZ + BUFSZ / BLK * REC_HDR + BLK)
/* Reads from the spill file are block-aligned, so each run of records can carry two blocks of slack. */
#define STAGE_SPILL ((size_t)MAX_PAGES * 3 * BLK + BLK)
#define MAX_PAGES (BUFSZ / BLK)
#define E_VALID (1ULL << 63)
#define E_OFF(e) ((e) & ((1ULL << 40) - 1))
#define E_LEN(e) ((uint32_t)((e) >> 40) & 0x1fff)
enum { SEG_FREE, SEG_OPEN, SEG_CLOSED, SEG_CLEANING };

struct rec {
	uint32_t lpn;
	uint16_t len, pad;
};

static struct {
	void *(*CreateCCtx)(void);
	void *(*CreateDCtx)(void);
	size_t (*Compress)(void *, void *, size_t, const void *, size_t, int);
	size_t (*Decompress)(void *, void *, size_t, const void *, size_t);
	unsigned (*IsError)(size_t);
} z;

static int clevel = 1;
static struct {
	pthread_mutex_t lock;
	uint64_t *map;	    /* per page: 0 reads as zeros, else E_VALID | length << 40 | offset of its record */
	uint32_t *live;	    /* per segment: bytes of records still current */
	uint32_t *used;	    /* per segment: bytes appended so far */
	uint8_t *state;
	uint64_t *seq;	    /* per segment: when it was opened, for finding the oldest */
	uint32_t *freelist, nseg, nfree, open;
	uint32_t *sfree, nspill, nsfree; /* spill segments are numbered nseg onwards */
	char *gcbuf, *evbuf, *rebuf;
	uint64_t clock;
	uint64_t pages, bytes, sbytes; /* pages held, the bytes their records take, and how much of that is spilled */
	uint64_t cleaned, refused, evicted, returned;
} cs = { .lock = PTHREAD_MUTEX_INITIALIZER };

static const char *spill_path;
static int spill_fd = -1;
static uint64_t spill_size;
static int high_pct = 90, low_pct = 70;
static void *rehydrate_thread(void *arg);

/* Per-thread compressor state and page-locked buffer for records on their way in or out. */
static __thread struct {
	void *cc, *dc;
	char *stage;
} ct;

static void ct_init(void)
{
	void *p;
	if (ct.stage)
		return;
	size_t sz = spill_fd >= 0 ? STAGE_SPILL : STAGE_BYTES;
	if (cu.MemAllocHost(&p, sz) && posix_memalign(&p, BLK, sz))
		die("cannot allocate staging buffer");
	ct.stage = p;
	ct.cc = z.CreateCCtx();
	ct.dc = z.CreateDCtx();
	if (!ct.cc || !ct.dc)
		die("cannot create zstd contexts");
}

static void cs_init(void)
{
	void *lib = dlopen("libzstd.so.1", RTLD_NOW), *p;
	if (!lib)
		die("--compress needs libzstd.so.1: %s", dlerror());
	z.CreateCCtx = sym(lib, "ZSTD_createCCtx");
	z.CreateDCtx = sym(lib, "ZSTD_createDCtx");
	z.Compress = sym(lib, "ZSTD_compressCCtx");
	z.Decompress = sym(lib, "ZSTD_decompressDCtx");
	z.IsError = sym(lib, "ZSTD_isError");

	cs.nseg = phys_size / SEG;
	if (spill_path) {
		/* O_DIRECT: nothing of ours should sit in the page cache while memory is short. */
		spill_fd = open(spill_path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC | O_DIRECT, 0600);
		if (spill_fd < 0 && errno == EINVAL)
			spill_fd = open(spill_path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
		if (spill_fd < 0)
			die("--spill %s: %s", spill_path, strerror(errno));
		if (!spill_size)
			spill_size = dev_size;
		int r = posix_fallocate(spill_fd, 0, spill_size);
		if (r)
			die("--spill %s: cannot reserve %llu MiB: %s", spill_path,
			    (unsigned long long)(spill_size / MiB), strerror(r));
		cs.nspill = spill_size / SEG;
		fprintf(stderr, "vroom: spilling to %s (%llu MiB) above %d%% full, back below %d%%\n",
			spill_path, (unsigned long long)(spill_size / MiB), high_pct, low_pct);
	}
	uint32_t all = cs.nseg + cs.nspill;
	cs.map = calloc(dev_size / BLK, sizeof *cs.map);
	cs.live = calloc(all, sizeof *cs.live);
	cs.used = calloc(all, sizeof *cs.used);
	cs.seq = calloc(all, sizeof *cs.seq);
	cs.state = calloc(all, 1);
	cs.freelist = calloc(cs.nseg, sizeof *cs.freelist);
	cs.sfree = calloc(cs.nspill + 1, sizeof *cs.sfree);
	if (cu.MemAllocHost(&p, SEG) && posix_memalign(&p, BLK, SEG))
		p = NULL;
	cs.gcbuf = p;
	if (posix_memalign(&p, BLK, SEG))
		p = NULL;
	cs.evbuf = p;
	if (cu.MemAllocHost(&p, SEG) && posix_memalign(&p, BLK, SEG))
		p = NULL;
	cs.rebuf = p;
	if (!cs.map || !cs.live || !cs.used || !cs.seq || !cs.state || !cs.freelist || !cs.sfree ||
	    !cs.gcbuf || !cs.evbuf || !cs.rebuf)
		die("out of memory for the compressed store's tables");
	for (uint32_t s = cs.nseg; s-- > 1;)
		cs.freelist[cs.nfree++] = s;
	for (uint32_t s = cs.nspill; s-- > 0;)
		cs.sfree[cs.nsfree++] = cs.nseg + s;
	cs.open = 0;
	cs.state[0] = SEG_OPEN;
	if (spill_fd >= 0) {
		pthread_t t;
		pthread_create(&t, NULL, rehydrate_thread, NULL);
	}
}

static void seg_free(uint32_t s)
{
	cs.state[s] = SEG_FREE;
	cs.used[s] = 0;
	if (s < cs.nseg)
		cs.freelist[cs.nfree++] = s;
	else
		cs.sfree[cs.nsfree++] = s;
}

/* Forget the stored copy of a page, if there is one. */
static void map_drop(uint64_t lpn)
{
	uint64_t e = cs.map[lpn];
	if (!e)
		return;
	uint32_t s = E_OFF(e) / SEG, sz = REC_HDR + E_LEN(e);
	cs.map[lpn] = 0;
	cs.pages--;
	cs.bytes -= sz;
	if (s >= cs.nseg)
		cs.sbytes -= sz;
	cs.live[s] -= sz;
	if (!cs.live[s] && cs.state[s] == SEG_CLOSED)
		seg_free(s);
}

/* Append records (header, data, header, data...) to the log and point their pages at them. */
static int log_append(const char *buf, size_t bytes)
{
	while (bytes) {
		uint32_t s = cs.open;
		size_t room = SEG - cs.used[s], n = 0;
		struct rec r;

		while (n < bytes) {
			memcpy(&r, buf + n, sizeof r);
			if (n + REC_HDR + r.len > room)
				break;
			n += REC_HDR + r.len;
		}
		if (!n) { /* this segment is full: move to a free one */
			if (!cs.nfree)
				return ENOSPC;
			cs.state[s] = SEG_CLOSED;
			if (!cs.live[s])
				seg_free(s);
			cs.open = s = cs.freelist[--cs.nfree];
			cs.state[s] = SEG_OPEN;
			cs.seq[s] = ++cs.clock;
			continue;
		}
		uint64_t at = (uint64_t)s * SEG + cs.used[s];
		if (vram_rw(1, at, (char *)buf, n))
			return EIO;
		for (size_t pos = 0; pos < n; pos += REC_HDR + r.len) {
			memcpy(&r, buf + pos, sizeof r);
			map_drop(r.lpn);
			cs.map[r.lpn] = E_VALID | (uint64_t)r.len << 40 | (at + pos);
			cs.pages++;
			cs.bytes += REC_HDR + r.len;
			cs.live[s] += REC_HDR + r.len;
		}
		cs.used[s] += n;
		buf += n, bytes -= n;
	}
	return 0;
}

/*
 * Copy the records of segment v that are still current from src, which holds
 * the segment, to dst at offset out, as long as they fit below limit. Returns
 * the new offset; *all says whether every one of them fitted. dst may be src.
 */
static size_t survivors(const char *src, uint32_t v, char *dst, size_t out, size_t limit, int *all)
{
	struct rec r;

	if (all)
		*all = 1;
	for (size_t pos = 0; pos < cs.used[v]; pos += REC_HDR + r.len) {
		memcpy(&r, src + pos, sizeof r);
		if (!cs.map[r.lpn] || E_OFF(cs.map[r.lpn]) != (uint64_t)v * SEG + pos)
			continue;
		if (out + REC_HDR + r.len > limit) {
			if (all)
				*all = 0;
			break;
		}
		memmove(dst + out, src + pos, REC_HDR + r.len);
		out += REC_HDR + r.len;
	}
	return out;
}

static int spill_io(int wr, char *buf, size_t len, uint64_t off)
{
	while (len) {
		ssize_t n = wr ? pwrite(spill_fd, buf, len, off) : pread(spill_fd, buf, len, off);
		if (n <= 0) {
			if (n < 0 && errno == EINTR)
				continue;
			fprintf(stderr, "vroom: spill file %s failed: %s\n", wr ? "write" : "read",
				n ? strerror(errno) : "short");
			return -1;
		}
		buf += n, len -= n, off += n;
	}
	return 0;
}

/*
 * Move up to one segment's worth of the oldest live records from VRAM to the
 * spill file. If the file has no free segment, its emptiest one is rewritten
 * with its own survivors first. Fails with ENOSPC when the file is full too.
 */
static int evict(void)
{
	uint32_t slot, v, best = UINT32_MAX;
	uint64_t floor = 0;
	size_t out = 0;
	int reuse = !cs.nsfree, all;
	struct rec r;

	if (spill_fd < 0)
		return ENOSPC;
	if (reuse) {
		slot = 0;
		for (uint32_t s = cs.nseg; s < cs.nseg + cs.nspill; s++)
			if (cs.state[s] == SEG_CLOSED && cs.live[s] < best)
				best = cs.live[s], slot = s;
		if (best > SEG - SEG / 16)
			return ENOSPC;
		if (spill_io(0, cs.gcbuf, (cs.used[slot] + BLK - 1) & ~(size_t)(BLK - 1),
			     (uint64_t)(slot - cs.nseg) * SEG))
			return EIO;
		out = survivors(cs.gcbuf, slot, cs.evbuf, 0, SEG, NULL);
	} else {
		slot = cs.sfree[cs.nsfree - 1];
	}
	size_t own = out;
	for (;;) {
		uint64_t oldest = UINT64_MAX;
		v = 0;
		for (uint32_t s = 0; s < cs.nseg; s++)
			if (cs.state[s] == SEG_CLOSED && cs.seq[s] > floor && cs.seq[s] < oldest)
				oldest = cs.seq[s], v = s;
		if (oldest == UINT64_MAX)
			break;
		if (vram_rw(0, (uint64_t)v * SEG, cs.gcbuf, cs.used[v]))
			return EIO;
		out = survivors(cs.gcbuf, v, cs.evbuf, out, SEG, &all);
		if (!all)
			break;
		floor = oldest;
	}
	if (out == own)
		return ENOSPC; /* nothing in VRAM that can move */
	memset(cs.evbuf + out, 0, -out & (BLK - 1));
	if (spill_io(1, cs.evbuf, (out + BLK - 1) & ~(size_t)(BLK - 1), (uint64_t)(slot - cs.nseg) * SEG))
		return EIO;
	if (!reuse)
		cs.nsfree--;
	cs.state[slot] = SEG_CLEANING; /* dropping its old records must not free it */
	for (size_t pos = 0; pos < out; pos += REC_HDR + r.len) {
		memcpy(&r, cs.evbuf + pos, sizeof r);
		map_drop(r.lpn);
		cs.map[r.lpn] = E_VALID | (uint64_t)r.len << 40 | ((uint64_t)slot * SEG + pos);
		cs.pages++;
		cs.bytes += REC_HDR + r.len;
		cs.sbytes += REC_HDR + r.len;
		cs.live[slot] += REC_HDR + r.len;
	}
	cs.used[slot] = out;
	cs.seq[slot] = ++cs.clock;
	cs.state[slot] = SEG_CLOSED;
	cs.evicted += out - own;
	return 0;
}

/* Below the low-water mark, bring the most recently spilled segment back into VRAM. */
static void *rehydrate_thread(void *arg)
{
	(void)arg;
	cu.CtxSetCurrent(ctx);
	prctl(PR_SET_IO_FLUSHER, 1, 0, 0, 0);
	for (;;) {
		int did = 0;
		pthread_mutex_lock(&cs.lock);
		if (cs.sbytes && (cs.bytes - cs.sbytes) * 100 < phys_size * low_pct && cs.nfree >= 4) {
			uint32_t v = 0;
			uint64_t newest = 0;
			for (uint32_t s = cs.nseg; s < cs.nseg + cs.nspill; s++)
				if (cs.state[s] == SEG_CLOSED && cs.seq[s] > newest)
					newest = cs.seq[s], v = s;
			if (newest && !spill_io(0, cs.rebuf, (cs.used[v] + BLK - 1) & ~(size_t)(BLK - 1),
						(uint64_t)(v - cs.nseg) * SEG)) {
				size_t out = survivors(cs.rebuf, v, cs.rebuf, 0, SEG, NULL);
				if (!log_append(cs.rebuf, out)) {
					cs.returned += out;
					did = 1;
				}
			}
		}
		pthread_mutex_unlock(&cs.lock);
		if (!did)
			usleep(50000);
	}
	return NULL;
}

/* Empty the closed segment holding the least live data. Fails if none would give much back. */
static int clean_one(void)
{
	uint32_t v = 0, best = UINT32_MAX;
	size_t out = 0;

	for (uint32_t s = 0; s < cs.nseg; s++)
		if (cs.state[s] == SEG_CLOSED && cs.live[s] < best)
			best = cs.live[s], v = s;
	if (best > SEG - SEG / 16)
		return ENOSPC;
	if (vram_rw(0, (uint64_t)v * SEG, cs.gcbuf, cs.used[v]))
		return EIO;
	out = survivors(cs.gcbuf, v, cs.gcbuf, 0, SEG, NULL);
	cs.state[v] = SEG_CLEANING; /* keep it out of the free list until its records have moved */
	int err = log_append(cs.gcbuf, out);
	if (err) {
		cs.state[v] = SEG_CLOSED;
		return err;
	}
	seg_free(v);
	cs.cleaned++;
	return 0;
}

/*
 * Make room for `need` bytes of records, keeping two segments back for
 * cleaning to work in. Above the high-water mark every write also pays for
 * one eviction, so the card drains at the rate it is being filled.
 */
static int make_room(size_t need)
{
	int over = spill_fd >= 0 && (cs.bytes - cs.sbytes) * 100 > phys_size * high_pct;

	for (;;) {
		uint64_t avail = (uint64_t)cs.nfree * SEG + SEG - cs.used[cs.open];
		if (avail >= need + 2 * SEG) {
			if (over)
				evict(); /* if the file is full there is still room here */
			return 0;
		}
		int err = over ? evict() : clean_one();
		if (err == ENOSPC)
			err = over ? clean_one() : evict();
		if (err) {
			cs.refused += err == ENOSPC;
			return err;
		}
	}
}

static int page_is_zero(const char *p)
{
	uint64_t w;
	memcpy(&w, p, 8);
	return !w && !memcmp(p, p + 8, BLK - 8);
}

static int cs_write(uint64_t off, const char *buf, size_t len)
{
	uint64_t lpn = off / BLK, zero[MAX_PAGES / 64] = { 0 };
	size_t n = len / BLK, out = 0;
	int err = 0;

	if (off % BLK || len % BLK || n > MAX_PAGES)
		return EINVAL;
	ct_init();
	for (size_t i = 0; i < n; i++) {
		const char *p = buf + i * BLK;
		if (page_is_zero(p)) {
			zero[i / 64] |= 1ULL << (i % 64);
			continue;
		}
		size_t c = z.Compress(ct.cc, ct.stage + out + REC_HDR, RAW_ABOVE, p, BLK, clevel);
		if (z.IsError(c) || !c || c > RAW_ABOVE) {
			memcpy(ct.stage + out + REC_HDR, p, BLK);
			c = BLK;
		}
		struct rec r = { .lpn = lpn + i, .len = c };
		memcpy(ct.stage + out, &r, sizeof r);
		out += REC_HDR + c;
	}
	pthread_mutex_lock(&cs.lock);
	if (out && !(err = make_room(out)))
		err = log_append(ct.stage, out);
	if (!err)
		for (size_t i = 0; i < n; i++)
			if (zero[i / 64] >> (i % 64) & 1)
				map_drop(lpn + i);
	pthread_mutex_unlock(&cs.lock);
	return err;
}

static int cs_read(uint64_t off, char *buf, size_t len)
{
	uint64_t lpn = off / BLK;
	size_t n = len / BLK, in = 0;
	uint32_t pos[MAX_PAGES], clen[MAX_PAGES];

	if (off % BLK || len % BLK || n > MAX_PAGES)
		return EINVAL;
	ct_init();
	pthread_mutex_lock(&cs.lock);
	for (size_t i = 0; i < n;) {
		uint64_t e = cs.map[lpn + i], start, end;
		if (!e) {
			clen[i++] = 0;
			continue;
		}
		/* take every following page whose record comes straight after, in one copy */
		size_t first = i;
		start = E_OFF(e), end = start;
		do {
			pos[i] = in + (end - start) + REC_HDR;
			clen[i] = E_LEN(e);
			end += REC_HDR + E_LEN(e);
		} while (++i < n && (e = cs.map[lpn + i]) && E_OFF(e) == end && end != phys_size);
		int bad;
		if (start < phys_size) {
			bad = vram_rw(0, start, ct.stage + in, end - start);
			in += end - start;
		} else { /* in the spill file: whole blocks, into a block-aligned place */
			uint64_t fo = start - phys_size, a = fo & ~(uint64_t)(BLK - 1);
			size_t al = (in + BLK - 1) & ~(size_t)(BLK - 1), shift = al + (fo - a) - in;
			size_t span = ((fo + (end - start) + BLK - 1) & ~(uint64_t)(BLK - 1)) - a;
			for (size_t k = first; k < i; k++)
				pos[k] += shift;
			bad = spill_io(0, ct.stage + al, span, a);
			in = al + span;
		}
		if (bad) {
			pthread_mutex_unlock(&cs.lock);
			return EIO;
		}
	}
	pthread_mutex_unlock(&cs.lock);
	for (size_t i = 0; i < n; i++) {
		char *p = buf + i * BLK;
		if (!clen[i])
			memset(p, 0, BLK);
		else if (clen[i] == BLK)
			memcpy(p, ct.stage + pos[i], BLK);
		else if (z.Decompress(ct.dc, p, BLK, ct.stage + pos[i], clen[i]) != BLK) {
			fprintf(stderr, "vroom: stored page %llu does not decompress\n",
				(unsigned long long)(lpn + i));
			return EIO;
		}
	}
	return 0;
}

/* Discard: the pages read as zeros from now on and their space is given back. */
static int cs_discard(uint64_t off, uint64_t len)
{
	uint64_t first = (off + BLK - 1) / BLK, end = (off + len) / BLK;
	pthread_mutex_lock(&cs.lock);
	for (uint64_t p = first; p < end; p++)
		map_drop(p);
	pthread_mutex_unlock(&cs.lock);
	return 0;
}

/*
 * ---- counting repeats (VROOM_COUNT=1 in the environment) ----
 *
 * How often is a page written whose contents the device already holds? Every
 * page written is hashed and compared with what was last written to each
 * block. Nothing is done with the answer but count it, into
 * /run/vroom/DEVICE.count: it says what keeping one copy would save.
 */
static int counting;
static struct {
	pthread_mutex_t lock;
	uint64_t *last;	  /* per block: hash of what was last written, 0 if nothing */
	uint8_t *readback; /* per block: read since it was written */
	struct slot {
		uint64_t hash;
		uint32_t blk, n; /* a block that holds it, and how many do */
	} *tab;
	uint64_t cap, filled;
	uint64_t writes, zero, same_block, bounced, duplicate, fresh, reads;
} cn = { .lock = PTHREAD_MUTEX_INITIALIZER };

static uint64_t page_hash(const char *p)
{
	uint64_t h[4] = { 0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL, 0xD6E8FEB86659FD93ULL };
	for (size_t i = 0; i < BLK; i += 32) {
		uint64_t w[4];
		memcpy(w, p + i, 32);
		for (int k = 0; k < 4; k++)
			h[k] = (h[k] ^ w[k]) * 0x100000001B3ULL, h[k] ^= h[k] >> 29;
	}
	uint64_t x = h[0] ^ h[1] * 3 ^ h[2] * 5 ^ h[3] * 7;
	x ^= x >> 32, x *= 0xD6E8FEB86659FD93ULL, x ^= x >> 32;
	return x ? x : 1;
}

static struct slot *cn_find(uint64_t h)
{
	uint64_t i = h % cn.cap;
	while (cn.tab[i].hash && cn.tab[i].hash != h)
		i = i + 1 == cn.cap ? 0 : i + 1;
	return &cn.tab[i];
}

static void cn_add(uint64_t h, uint32_t blk)
{
	struct slot *s = cn_find(h);
	if (!s->hash)
		s->hash = h, cn.filled++;
	s->n++, s->blk = blk;
}

static void cn_init(void)
{
	uint64_t nblk = dev_size / BLK;
	cn.cap = nblk * 3 + 64;
	cn.last = calloc(nblk, sizeof *cn.last);
	cn.readback = calloc(nblk, 1);
	cn.tab = calloc(cn.cap, sizeof *cn.tab);
	if (!cn.last || !cn.readback || !cn.tab)
		die("out of memory for the repeat counters");
}

static void count_io(int wr, uint64_t off, const char *buf, size_t len)
{
	uint64_t blk = off / BLK;

	pthread_mutex_lock(&cn.lock);
	for (size_t i = 0; i < len / BLK; i++, blk++) {
		if (!wr) {
			cn.reads++;
			cn.readback[blk] = 1;
			continue;
		}
		const char *p = buf + i * BLK;
		uint64_t h = page_hash(p), old = cn.last[blk];
		struct slot *s;

		cn.writes++;
		if (page_is_zero(p))
			cn.zero++;
		else if (h == old)
			cn.same_block++;
		else if ((s = cn_find(h))->hash && s->n) /* some other block holds exactly this */
			cn.last[s->blk] == h && cn.readback[s->blk] ? cn.bounced++ : cn.duplicate++;
		else
			cn.fresh++;
		if (h != old) {
			if (old)
				cn_find(old)->n--;
			cn_add(h, blk);
			cn.last[blk] = h;
		}
		cn.readback[blk] = 0;
		if (cn.filled * 3 > cn.cap * 2) { /* too many hashes no block holds any more: start again */
			memset(cn.tab, 0, cn.cap * sizeof *cn.tab);
			cn.filled = 0;
			for (uint64_t b = 0; b < dev_size / BLK; b++)
				if (cn.last[b])
					cn_add(cn.last[b], b);
		}
	}
	pthread_mutex_unlock(&cn.lock);
}

static void count_print(FILE *f)
{
	fprintf(f, "WRITES=%llu\nREADS=%llu\nZERO=%llu\nSAME_BLOCK=%llu\nBOUNCED=%llu\nDUPLICATE=%llu\nFRESH=%llu\n",
		(unsigned long long)cn.writes, (unsigned long long)cn.reads, (unsigned long long)cn.zero,
		(unsigned long long)cn.same_block, (unsigned long long)cn.bounced,
		(unsigned long long)cn.duplicate, (unsigned long long)cn.fresh);
}

/* One read or write of the device, whichever way it is stored. Returns 0 or an errno. */
static int dev_rw(int wr, uint64_t off, char *buf, size_t len)
{
	if (counting && !(off % BLK))
		count_io(wr, off, buf, len);
	if (!cratio)
		return vram_rw(wr, off, buf, len) ? EIO : 0;
	return wr ? cs_write(off, buf, len) : cs_read(off, buf, len);
}

static const char *dev_path;
static void *count_thread(void *arg)
{
	char path[128], tmp[136];
	const char *base = strrchr(dev_path, '/');

	(void)arg;
	mkdir("/run/vroom", 0755);
	snprintf(path, sizeof path, "/run/vroom/%s.count", base ? base + 1 : dev_path);
	snprintf(tmp, sizeof tmp, "%s.tmp", path);
	for (;; sleep(10)) {
		FILE *f = fopen(tmp, "w");
		if (!f)
			continue;
		pthread_mutex_lock(&cn.lock);
		count_print(f);
		pthread_mutex_unlock(&cn.lock);
		fclose(f);
		rename(tmp, path);
	}
	return NULL;
}

/* While serving: keep /run/vroom/DEVICE.stat current for 'vroomctl list', and warn when nearly full. */
static void *stat_thread(void *arg)
{
	char path[128], tmp[136];
	const char *base = strrchr(dev_path, '/');
	uint64_t last = ~0ULL;
	int warned = 0;

	(void)arg;
	mkdir("/run/vroom", 0755);
	snprintf(path, sizeof path, "/run/vroom/%s.stat", base ? base + 1 : dev_path);
	snprintf(tmp, sizeof tmp, "%s.tmp", path);
	for (;; sleep(2)) {
		pthread_mutex_lock(&cs.lock);
		uint64_t pages = cs.pages, bytes = cs.bytes - cs.sbytes, refused = cs.refused, sbytes = cs.sbytes;
		uint64_t held = (uint64_t)(cs.nseg - cs.nfree) * SEG;
		pthread_mutex_unlock(&cs.lock);
		int pct = bytes * 100 / phys_size;

		if (spill_fd >= 0)
			pct = spill_size ? sbytes * 100 / spill_size : 0;
		if (pct >= 90 && !warned)
			fprintf(stderr, "vroom: WARNING: the %s is %d%% full of compressed data; "
					"writes will fail when it is full\n",
				spill_fd >= 0 ? "spill file" : "VRAM", pct);
		warned = pct >= 90 ? 1 : pct < 80 ? 0 : warned;
		if (pages + bytes + refused + sbytes * 3 == last)
			continue;
		last = pages + bytes + refused + sbytes * 3;
		FILE *f = fopen(tmp, "w");
		if (!f)
			continue;
		fprintf(f, "PAGES=%llu\nSTORED_BYTES=%llu\nSEGMENT_BYTES=%llu\nVRAM_BYTES=%llu\nREFUSED=%llu\n"
			   "SPILLED_BYTES=%llu\nSPILL_BYTES=%llu\n",
			(unsigned long long)pages, (unsigned long long)bytes, (unsigned long long)held,
			(unsigned long long)phys_size, (unsigned long long)refused,
			(unsigned long long)sbytes, (unsigned long long)spill_size);
		fclose(f);
		rename(tmp, path);
	}
	return NULL;
}

/* ---- NBD transmission protocol ---- */
struct nbd_req {
	uint32_t magic, type;
	uint64_t cookie, from;
	uint32_t len;
} __attribute__((packed));
struct nbd_rep {
	uint32_t magic, error;
	uint64_t cookie;
} __attribute__((packed));

static int rd_full(int fd, void *p, size_t n)
{
	while (n) {
		ssize_t r = recv(fd, p, n, 0);
		if (r <= 0) {
			if (r < 0 && errno == EINTR)
				continue;
			return -1;
		}
		p = (char *)p + r, n -= r;
	}
	return 0;
}

static int wr_full(int fd, const void *p, size_t n, int flags)
{
	while (n) {
		ssize_t r = send(fd, p, n, flags | MSG_NOSIGNAL);
		if (r < 0) {
			if (errno == EINTR)
				continue;
			return -1;
		}
		p = (const char *)p + r, n -= r;
	}
	return 0;
}

static void *worker(void *arg)
{
	int fd = (intptr_t)arg;
	void *vbuf;
	cu.CtxSetCurrent(ctx);
	/* Page-locked staging memory: fast copies, and nothing to pin per I/O. */
	if (cu.MemAllocHost(&vbuf, BUFSZ) && posix_memalign(&vbuf, BLK, BUFSZ))
		die("cannot allocate staging buffer");
	char *buf = vbuf;
	/* We sit in the swap-out path: our allocations must not wait on I/O. */
	prctl(PR_SET_IO_FLUSHER, 1, 0, 0, 0);

	for (;;) {
		struct nbd_req rq;
		if (rd_full(fd, &rq, sizeof rq))
			break;
		if (be32toh(rq.magic) != NBD_REQUEST_MAGIC) {
			fprintf(stderr, "vroom: bad request magic\n");
			break;
		}
		uint32_t cmd = be32toh(rq.type) & 0xffff, len = be32toh(rq.len);
		uint64_t off = be64toh(rq.from);
		struct nbd_rep rp = { htobe32(NBD_REPLY_MAGIC), 0, rq.cookie };
		int err = off + len > dev_size ? EINVAL : 0;
		size_t n;

		switch (cmd) {
		case NBD_CMD_READ:
			n = len < BUFSZ ? len : BUFSZ;
			if (!err)
				err = dev_rw(0, off, buf, n);
			if (err)
				break;
			if (wr_full(fd, &rp, sizeof rp, MSG_MORE))
				goto out;
			for (;;) {
				if (wr_full(fd, buf, n, 0))
					goto out;
				off += n, len -= n;
				if (!len)
					break;
				n = len < BUFSZ ? len : BUFSZ;
				if (dev_rw(0, off, buf, n))
					goto out; /* mid-reply: drop the connection */
			}
			continue;
		case NBD_CMD_WRITE:
			while (len) {
				n = len < BUFSZ ? len : BUFSZ;
				if (rd_full(fd, buf, n))
					goto out;
				if (!err)
					err = dev_rw(1, off, buf, n);
				off += n, len -= n;
			}
			break;
		case NBD_CMD_FLUSH:
			err = 0;
			break;
		case NBD_CMD_TRIM:
			if (!err && cratio)
				err = cs_discard(off, len);
			break;
		case NBD_CMD_DISC:
			goto out;
		default:
			err = EINVAL;
		}
		rp.error = htobe32(err);
		if (wr_full(fd, &rp, sizeof rp, 0))
			break;
	}
out:
	close(fd);
	return NULL;
}

/* ---- kernel NBD attachment ---- */
static int nbd_fd = -1;

/* Is our device listed (first column) in /proc/swaps or /proc/mounts? */
static int listed_in(const char *file)
{
	char line[512];
	size_t n = strlen(dev_path);
	int hit = 0;
	FILE *f = fopen(file, "r");
	if (!f)
		return 0;
	while (fgets(line, sizeof line, f))
		if (!strncmp(line, dev_path, n) && (line[n] == ' ' || line[n] == '\t'))
			hit = 1;
	fclose(f);
	return hit;
}

/* Returns once a stop signal arrives while the device holds nothing live. */
static void *sig_thread(void *arg)
{
	sigset_t *set = arg;
	int sig;
	for (;;) {
		sigwait(set, &sig);
		int sw = listed_in("/proc/swaps");
		if (sw || listed_in("/proc/mounts")) {
			fprintf(stderr, "vroom: ignoring signal %d: %s is %s. Run '%s %s' first.\n",
				sig, dev_path, sw ? "active swap" : "mounted",
				sw ? "swapoff" : "umount", dev_path);
			continue;
		}
		if (nbd_fd >= 0)
			ioctl(nbd_fd, NBD_DISCONNECT);
		return NULL;
	}
}

static sigset_t stop_set;
static void block_stop_signals(void)
{
	sigemptyset(&stop_set);
	sigaddset(&stop_set, SIGINT);
	sigaddset(&stop_set, SIGTERM);
	pthread_sigmask(SIG_BLOCK, &stop_set, NULL);
}

/* We sit in the kernel's swap-out path: stay resident and never be the OOM victim. */
static void harden(void)
{
	if (mlockall(MCL_CURRENT | MCL_FUTURE))
		fprintf(stderr, "vroom: warning: mlockall: %s\n", strerror(errno));
	if (prctl(PR_SET_IO_FLUSHER, 1, 0, 0, 0))
		fprintf(stderr, "vroom: warning: PR_SET_IO_FLUSHER: %s\n", strerror(errno));
	FILE *f = fopen("/proc/self/oom_score_adj", "w");
	if (f) {
		fputs("-1000", f);
		fclose(f);
	}
}

static void serve_nbd(int threads)
{
	pthread_t t;

	nbd_fd = open(dev_path, O_RDWR);
	if (nbd_fd < 0)
		die("open %s: %s (modprobe nbd? root?)", dev_path, strerror(errno));

	/* Never touch a device someone else (e.g. nbd-client) has attached. */
	char pidpath[128];
	const char *base = strrchr(dev_path, '/');
	snprintf(pidpath, sizeof pidpath, "/sys/block/%s/pid", base ? base + 1 : dev_path);
	if (!access(pidpath, F_OK))
		die("%s is already attached; pick another nbd device", dev_path);

	if (ioctl(nbd_fd, NBD_SET_BLKSIZE, (unsigned long)BLK) ||
	    ioctl(nbd_fd, NBD_SET_SIZE_BLOCKS, (unsigned long)(dev_size / BLK)) ||
	    ioctl(nbd_fd, NBD_SET_FLAGS,
		  (unsigned long)(NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH | NBD_FLAG_CAN_MULTI_CONN |
				  (cratio ? NBD_FLAG_SEND_TRIM : 0))))
		die("configuring %s: %s", dev_path, strerror(errno));

	for (int i = 0; i < threads; i++) {
		int sp[2];
		if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp))
			die("socketpair: %s", strerror(errno));
		if (ioctl(nbd_fd, NBD_SET_SOCK, sp[0]))
			die("NBD_SET_SOCK: %s (device busy?)", strerror(errno));
		pthread_create(&t, NULL, worker, (void *)(intptr_t)sp[1]);
		close(sp[0]); /* the kernel holds its own reference */
	}
	pthread_create(&t, NULL, sig_thread, &stop_set);
	if (cratio)
		pthread_create(&t, NULL, stat_thread, NULL);
	if (counting)
		pthread_create(&t, NULL, count_thread, NULL);
	harden();

	fprintf(stderr, "vroom: serving %llu MiB on %s (nbd, %d threads)\n",
		(unsigned long long)(dev_size / MiB), dev_path, threads);
	ioctl(nbd_fd, NBD_DO_IT); /* blocks until disconnect */
	ioctl(nbd_fd, NBD_CLEAR_QUE);
	ioctl(nbd_fd, NBD_CLEAR_SOCK);
	fprintf(stderr, "vroom: %s disconnected, releasing VRAM\n", dev_path);
}

/* ---- ublk backend: requests arrive over io_uring, no socket in between ---- */
struct ring {
	int fd, shift;
	unsigned *sq_tail, *sq_mask, *sq_array, *cq_head, *cq_tail, *cq_mask;
	unsigned tail, pending;
	char *sqes;
	struct io_uring_cqe *cqes;
};

static void ring_init(struct ring *r, unsigned entries, unsigned flags)
{
	struct io_uring_params p = { .flags = flags };
	r->fd = syscall(__NR_io_uring_setup, entries, &p);
	if (r->fd < 0)
		die("io_uring_setup: %s", strerror(errno));
	if (!(p.features & IORING_FEAT_SINGLE_MMAP))
		die("kernel too old (io_uring single mmap)");
	size_t sq = p.sq_off.array + p.sq_entries * sizeof(unsigned);
	size_t cq = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
	r->shift = flags & IORING_SETUP_SQE128 ? 7 : 6;
	char *m = mmap(NULL, sq > cq ? sq : cq, PROT_READ | PROT_WRITE,
		       MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_SQ_RING);
	r->sqes = mmap(NULL, (size_t)p.sq_entries << r->shift, PROT_READ | PROT_WRITE,
		       MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_SQES);
	if (m == MAP_FAILED || r->sqes == MAP_FAILED)
		die("io_uring mmap: %s", strerror(errno));
	r->sq_tail = (unsigned *)(m + p.sq_off.tail);
	r->sq_mask = (unsigned *)(m + p.sq_off.ring_mask);
	r->sq_array = (unsigned *)(m + p.sq_off.array);
	r->cq_head = (unsigned *)(m + p.cq_off.head);
	r->cq_tail = (unsigned *)(m + p.cq_off.tail);
	r->cq_mask = (unsigned *)(m + p.cq_off.ring_mask);
	r->cqes = (struct io_uring_cqe *)(m + p.cq_off.cqes);
	r->tail = *r->sq_tail;
	r->pending = 0;
}

/* Queue one IORING_OP_URING_CMD; the caller never queues more than the ring holds. */
static void ring_cmd(struct ring *r, int fd, unsigned op, const void *cmd, size_t len, uint64_t ud)
{
	unsigned idx = r->tail & *r->sq_mask;
	struct io_uring_sqe *s = (struct io_uring_sqe *)(r->sqes + ((size_t)idx << r->shift));
	memset(s, 0, (size_t)1 << r->shift);
	s->opcode = IORING_OP_URING_CMD;
	s->fd = fd;
	s->cmd_op = op;
	s->user_data = ud;
	memcpy((char *)s + offsetof(struct io_uring_sqe, cmd), cmd, len);
	r->sq_array[idx] = idx;
	r->tail++;
	r->pending++;
}

static int ring_enter(struct ring *r, unsigned wait)
{
	__atomic_store_n(r->sq_tail, r->tail, __ATOMIC_RELEASE);
	for (;;) {
		int n = syscall(__NR_io_uring_enter, r->fd, r->pending, wait,
				IORING_ENTER_GETEVENTS, NULL, 0);
		if (n >= 0) {
			r->pending -= n;
			return 0;
		}
		if (errno != EINTR)
			return -errno;
	}
}

#define UBLK_DEPTH 32
#define UBLK_IO_BYTES (256 * 1024)
static struct ring ctrl_ring;
static int ctrl_fd, cdev_fd;
static unsigned ublk_id, ublk_io_bytes;

static int ublk_ctrl(unsigned op, void *addr, unsigned len, uint64_t data0)
{
	struct ublksrv_ctrl_cmd c = { .dev_id = ublk_id, .queue_id = (uint16_t)-1, .len = len,
				      .addr = (uintptr_t)addr, .data = { data0 } };
	ring_cmd(&ctrl_ring, ctrl_fd, op, &c, sizeof c, 0);
	int r = ring_enter(&ctrl_ring, 1);
	if (r)
		return r;
	unsigned head = *ctrl_ring.cq_head;
	r = ctrl_ring.cqes[head & *ctrl_ring.cq_mask].res;
	__atomic_store_n(ctrl_ring.cq_head, head + 1, __ATOMIC_RELEASE);
	return r;
}

static void *ublk_queue(void *arg)
{
	int q = (intptr_t)arg, r;
	struct ring ring;
	char *buf[UBLK_DEPTH];

	cu.CtxSetCurrent(ctx);
	prctl(PR_SET_IO_FLUSHER, 1, 0, 0, 0);
	ring_init(&ring, UBLK_DEPTH,
		  IORING_SETUP_COOP_TASKRUN | IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN);

	size_t pg = sysconf(_SC_PAGESIZE);
	size_t dsz = (UBLK_DEPTH * sizeof(struct ublksrv_io_desc) + pg - 1) & ~(pg - 1);
	const struct ublksrv_io_desc *iod =
		mmap(NULL, dsz, PROT_READ, MAP_SHARED | MAP_POPULATE, cdev_fd,
		     UBLKSRV_CMD_BUF_OFFSET +
			     (off_t)q * UBLK_MAX_QUEUE_DEPTH * sizeof(struct ublksrv_io_desc));
	if (iod == MAP_FAILED)
		die("ublk: mapping queue %d descriptors: %s", q, strerror(errno));

	for (int tag = 0; tag < UBLK_DEPTH; tag++) {
		void *p;
		if (cu.MemAllocHost(&p, ublk_io_bytes) && posix_memalign(&p, BLK, ublk_io_bytes))
			die("cannot allocate I/O buffer");
		buf[tag] = p;
		struct ublksrv_io_cmd c = { .q_id = q, .tag = tag, .addr = (uintptr_t)p };
		ring_cmd(&ring, cdev_fd, UBLK_U_IO_FETCH_REQ, &c, sizeof c, tag);
	}

	for (unsigned live = UBLK_DEPTH; live;) {
		if ((r = ring_enter(&ring, 1)))
			die("ublk: io_uring_enter: %s", strerror(-r));
		unsigned head = *ring.cq_head, tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
		for (; head != tail; head++) {
			struct io_uring_cqe *e = &ring.cqes[head & *ring.cq_mask];
			unsigned tag = e->user_data;
			if (e->res != UBLK_IO_RES_OK) { /* device stopping: no re-fetch */
				if (e->res != UBLK_IO_RES_ABORT)
					fprintf(stderr, "vroom: ublk queue %d tag %u: %s\n", q, tag,
						strerror(-e->res));
				live--;
				continue;
			}
			const struct ublksrv_io_desc *d = &iod[tag];
			uint64_t off = d->start_sector << 9, len = (uint64_t)d->nr_sectors << 9;
			int res;
			switch (d->op_flags & 0xff) {
			case UBLK_IO_OP_READ:
			case UBLK_IO_OP_WRITE:
				if (len > ublk_io_bytes || off + len > dev_size)
					res = -EINVAL;
				else if ((res = -dev_rw((d->op_flags & 0xff) == UBLK_IO_OP_WRITE, off,
							buf[tag], len)) == 0)
					res = len;
				break;
			case UBLK_IO_OP_DISCARD:
			case UBLK_IO_OP_WRITE_ZEROES:
				if (!cratio || off + len > dev_size)
					res = -EOPNOTSUPP;
				else
					res = cs_discard(off, len) ? -EIO : (int)len;
				break;
			case UBLK_IO_OP_FLUSH:
				res = 0;
				break;
			default:
				res = -EOPNOTSUPP;
			}
			struct ublksrv_io_cmd c = { .q_id = q, .tag = tag, .result = res,
						    .addr = (uintptr_t)buf[tag] };
			ring_cmd(&ring, cdev_fd, UBLK_U_IO_COMMIT_AND_FETCH_REQ, &c, sizeof c, tag);
		}
		__atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
	}
	/* Drop our references to the char device, or DEL_DEV would wait forever. */
	munmap((void *)iod, dsz);
	close(ring.fd);
	return NULL;
}

static void serve_ublk(int threads)
{
	pthread_t tq[MAX_THREADS], ts;
	char path[64];
	int r;

	if (sscanf(dev_path, "/dev/ublkb%u", &ublk_id) != 1)
		die("ublk device must be named /dev/ublkbN");
	ctrl_fd = open("/dev/ublk-control", O_RDWR);
	if (ctrl_fd < 0)
		die("open /dev/ublk-control: %s (modprobe ublk_drv? root?)", strerror(errno));
	ring_init(&ctrl_ring, 4, IORING_SETUP_SQE128);

	struct ublksrv_ctrl_dev_info info, old;
	for (int attempt = 0;; attempt++) {
		info = (struct ublksrv_ctrl_dev_info){ .nr_hw_queues = threads,
						       .queue_depth = UBLK_DEPTH,
						       .max_io_buf_bytes = UBLK_IO_BYTES,
						       .dev_id = ublk_id,
						       .flags = UBLK_F_CMD_IOCTL_ENCODE };
		r = ublk_ctrl(UBLK_U_CMD_ADD_DEV, &info, sizeof info, 0);
		/* A dead device with our id is the remains of an earlier daemon: replace it. */
		if (r == -EEXIST && !attempt &&
		    !ublk_ctrl(UBLK_U_CMD_GET_DEV_INFO, &old, sizeof old, 0) &&
		    old.state == UBLK_S_DEV_DEAD && !ublk_ctrl(UBLK_U_CMD_DEL_DEV, NULL, 0, 0))
			continue;
		break;
	}
	if (r)
		die("ublk: cannot create %s: %s", dev_path, strerror(-r));
	ublk_io_bytes = info.max_io_buf_bytes;

	struct ublk_params p = { .len = sizeof p,
				 .types = UBLK_PARAM_TYPE_BASIC,
				 .basic = { .logical_bs_shift = 12,
					    .physical_bs_shift = 12,
					    .io_min_shift = 12,
					    .io_opt_shift = 12,
					    .max_sectors = ublk_io_bytes >> 9,
					    .dev_sectors = dev_size >> 9 },
				 .discard = { .discard_granularity = BLK,
					      .max_discard_sectors = 1 << 21,
					      .max_write_zeroes_sectors = 1 << 21,
					      .max_discard_segments = 1 } };
	if (cratio)
		p.types |= UBLK_PARAM_TYPE_DISCARD;
	if ((r = ublk_ctrl(UBLK_U_CMD_SET_PARAMS, &p, sizeof p, 0)))
		die("ublk: set params: %s", strerror(-r));

	snprintf(path, sizeof path, "/dev/ublkc%u", ublk_id);
	for (int i = 0; (cdev_fd = open(path, O_RDWR)) < 0; i++) {
		if (i == 50)
			die("open %s: %s", path, strerror(errno));
		usleep(100000); /* udev may still be creating the node */
	}

	for (int i = 0; i < threads; i++)
		pthread_create(&tq[i], NULL, ublk_queue, (void *)(intptr_t)i);
	/* Returns once every queue has handed all its tags to the kernel. */
	if ((r = ublk_ctrl(UBLK_U_CMD_START_DEV, NULL, 0, getpid())))
		die("ublk: start: %s", strerror(-r));
	harden();
	fprintf(stderr, "vroom: serving %llu MiB on %s (ublk, %d queues)\n",
		(unsigned long long)(dev_size / MiB), dev_path, threads);

	pthread_create(&ts, NULL, sig_thread, &stop_set);
	if (cratio || counting) {
		pthread_t tstat;
		if (cratio)
			pthread_create(&tstat, NULL, stat_thread, NULL);
		if (counting)
			pthread_create(&tstat, NULL, count_thread, NULL);
	}
	pthread_join(ts, NULL);
	ublk_ctrl(UBLK_U_CMD_STOP_DEV, NULL, 0, 0);
	for (int i = 0; i < threads; i++)
		pthread_join(tq[i], NULL);
	close(cdev_fd);
	ublk_ctrl(UBLK_U_CMD_DEL_DEV, NULL, 0, 0);
	fprintf(stderr, "vroom: %s removed, releasing VRAM\n", dev_path);
}

/* ---- read-only benchmark of any block device (safe on a live swap device) ---- */
static double now(void);
#define BENCH_OPS 20000
static uint64_t bench_blocks;
static void *bench_reader(void *arg)
{
	void *buf;
	uint64_t x = 88172645463325252ULL + (uintptr_t)arg;
	int fd = open(dev_path, O_RDONLY | O_DIRECT);
	if (fd < 0 || posix_memalign(&buf, BLK, BLK))
		die("open %s: %s", dev_path, strerror(errno));
	for (int i = 0; i < BENCH_OPS; i++) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		if (pread(fd, buf, BLK, x % bench_blocks * BLK) != BLK)
			die("bench: read failed: %s", strerror(errno));
	}
	close(fd);
	return NULL;
}

static void bench(int threads)
{
	pthread_t t[MAX_THREADS];
	uint64_t bytes;
	int fd = open(dev_path, O_RDONLY);
	if (fd < 0 || ioctl(fd, BLKGETSIZE64, &bytes) || bytes < BLK)
		die("%s: not a usable block device: %s", dev_path, strerror(errno));
	close(fd);
	bench_blocks = bytes / BLK;
	for (int n = 1; n <= threads; n += threads - 1 ? threads - 1 : 1) {
		double t0 = now();
		for (int i = 0; i < n; i++)
			pthread_create(&t[i], NULL, bench_reader, (void *)(intptr_t)i);
		for (int i = 0; i < n; i++)
			pthread_join(t[i], NULL);
		double dt = now() - t0;
		printf("%s 4K random read  %7.0f IOPS  (%d thread%s, %.1f us per read)\n", dev_path,
		       (double)BENCH_OPS * n / dt, n, n > 1 ? "s" : "", dt / BENCH_OPS * 1e6);
	}
}

/* ---- self-test: drives the same worker code as the kernel would ---- */
static double now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void fill(char *buf, size_t len, uint64_t seed)
{
	uint64_t x = seed * 0x9E3779B97F4A7C15ULL + 0x1234567;
	for (size_t i = 0; i < len; i += 8) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		memcpy(buf + i, &x, 8);
	}
}

static int nbd_io(int fd, int cmd, uint64_t off, char *buf, uint32_t len)
{
	static uint64_t cookie;
	struct nbd_req rq = { htobe32(NBD_REQUEST_MAGIC), htobe32(cmd),
			      __atomic_add_fetch(&cookie, 1, __ATOMIC_RELAXED), htobe64(off),
			      htobe32(len) };
	struct nbd_rep rp;
	if (wr_full(fd, &rq, sizeof rq, cmd == NBD_CMD_WRITE ? MSG_MORE : 0))
		return -1;
	if (cmd == NBD_CMD_WRITE && wr_full(fd, buf, len, 0))
		return -1;
	if (rd_full(fd, &rp, sizeof rp) || be32toh(rp.magic) != NBD_REPLY_MAGIC ||
	    rp.cookie != rq.cookie)
		return -1;
	if (rp.error)
		return be32toh(rp.error);
	if (cmd == NBD_CMD_READ && rd_full(fd, buf, len))
		return -1;
	return 0;
}

#define RAND_OPS 20000
static void *rand_reader(void *arg)
{
	int fd = (intptr_t)arg;
	char buf[BLK];
	uint64_t x = 88172645463325252ULL + fd;
	for (int i = 0; i < RAND_OPS; i++) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		if (nbd_io(fd, NBD_CMD_READ, x % (dev_size / BLK) * BLK, buf, BLK))
			die("selftest: parallel read failed");
	}
	return NULL;
}

static void selftest(int threads)
{
	int fds[MAX_THREADS];
	pthread_t tw[MAX_THREADS], tc[MAX_THREADS];
	char *a = malloc(BUFSZ * 2), *b = malloc(BUFSZ * 2);
	double t;

	for (int i = 0; i < threads; i++) {
		int sp[2];
		if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp))
			die("socketpair: %s", strerror(errno));
		fds[i] = sp[0];
		pthread_create(&tw[i], NULL, worker, (void *)(intptr_t)sp[1]);
	}
	int fd = fds[0];

	t = now();
	for (uint64_t off = 0; off < dev_size; off += BUFSZ) {
		fill(a, BUFSZ, off);
		if (nbd_io(fd, NBD_CMD_WRITE, off, a, BUFSZ))
			die("selftest: write failed at %llu", (unsigned long long)off);
	}
	printf("sequential write  %7.0f MiB/s\n", dev_size / MiB / (now() - t));

	t = now();
	for (uint64_t off = 0; off < dev_size; off += BUFSZ) {
		fill(a, BUFSZ, off);
		if (nbd_io(fd, NBD_CMD_READ, off, b, BUFSZ) || memcmp(a, b, BUFSZ))
			die("selftest: MISMATCH in sequential read at %llu", (unsigned long long)off);
	}
	printf("sequential read   %7.0f MiB/s  (verified)\n", dev_size / MiB / (now() - t));

	if (dev_size > CHUNK) { /* a request spanning two VRAM allocations, and >1 buffer */
		uint64_t off = CHUNK - BUFSZ - BLK;
		fill(a, 2 * BUFSZ, 42);
		if (nbd_io(fd, NBD_CMD_WRITE, off, a, 2 * BUFSZ) ||
		    nbd_io(fd, NBD_CMD_READ, off, b, 2 * BUFSZ) || memcmp(a, b, 2 * BUFSZ))
			die("selftest: MISMATCH across chunk boundary");
		printf("chunk-boundary I/O    ok\n");
	}
	if (nbd_io(fd, NBD_CMD_READ, dev_size, b, BLK) != EINVAL)
		die("selftest: out-of-range read was not rejected");

	uint64_t x = 2463534242ULL, nblk = dev_size / BLK;
	t = now();
	for (int i = 0; i < RAND_OPS; i++) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		uint64_t blk = x % nblk;
		fill(a, BLK, blk ^ 0xABCDEF);
		if (nbd_io(fd, NBD_CMD_WRITE, blk * BLK, a, BLK))
			die("selftest: 4K write failed");
	}
	double dw = now() - t;
	x = 2463534242ULL;
	t = now();
	for (int i = 0; i < RAND_OPS; i++) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		uint64_t blk = x % nblk;
		fill(a, BLK, blk ^ 0xABCDEF);
		if (nbd_io(fd, NBD_CMD_READ, blk * BLK, b, BLK) || memcmp(a, b, BLK))
			die("selftest: MISMATCH in 4K random read, block %llu", (unsigned long long)blk);
	}
	double dr = now() - t;
	printf("4K random write   %7.0f IOPS  (%.1f us each, 1 thread)\n", RAND_OPS / dw,
	       dw / RAND_OPS * 1e6);
	printf("4K random read    %7.0f IOPS  (%.1f us each, 1 thread, verified)\n",
	       RAND_OPS / dr, dr / RAND_OPS * 1e6);

	t = now();
	for (int i = 0; i < threads; i++)
		pthread_create(&tc[i], NULL, rand_reader, (void *)(intptr_t)fds[i]);
	for (int i = 0; i < threads; i++)
		pthread_join(tc[i], NULL);
	printf("4K random read    %7.0f IOPS  (%d threads)\n",
	       (double)RAND_OPS * threads / (now() - t), threads);

	/* The same copies with no socket or thread hop: the floor set by the driver. */
	void *pin;
	cu.CtxSetCurrent(ctx);
	if (cu.MemAllocHost(&pin, BLK))
		die("selftest: cannot allocate pinned buffer");
	x = 2463534242ULL;
	t = now();
	for (int i = 0; i < RAND_OPS; i++) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		if (vram_rw(0, x % nblk * BLK, pin, BLK))
			die("selftest: direct read failed");
	}
	dr = now() - t;
	printf("4K random read    %7.0f IOPS  (%.1f us each, direct VRAM copy, no socket)\n",
	       RAND_OPS / dr, dr / RAND_OPS * 1e6);

	if (counting) { /* the same page to three blocks, with a read in between: one duplicate, one bounce */
		uint64_t d = cn.duplicate, bo = cn.bounced;
		fill(a, BLK, 0xC0FFEE);
		if (nbd_io(fd, NBD_CMD_WRITE, 0, a, BLK) || nbd_io(fd, NBD_CMD_WRITE, BLK, a, BLK) ||
		    nbd_io(fd, NBD_CMD_READ, BLK, b, BLK) || nbd_io(fd, NBD_CMD_WRITE, 2 * BLK, a, BLK) ||
		    cn.duplicate != d + 1 || cn.bounced != bo + 1)
			die("selftest: repeat counters are wrong");
		printf("repeat counters       ok\n");
	}
	for (int i = 0; i < threads; i++) {
		struct nbd_req rq = { htobe32(NBD_REQUEST_MAGIC), htobe32(NBD_CMD_DISC), 0, 0, 0 };
		wr_full(fds[i], &rq, sizeof rq, 0);
		pthread_join(tw[i], NULL);
	}
	printf("selftest passed\n");
}

/* ---- self-test of the compressed store ---- */

/* Page contents that depend on the block and how often it has been rewritten: one in
 * eight all zeros, one in eight incompressible, the rest part noise and part repetition. */
static void pattern(char *p, uint64_t blk, unsigned ver)
{
	uint64_t x = (blk + 1) * 0x9E3779B97F4A7C15ULL ^ (ver + 1) * 0xD6E8FEB86659FD93ULL;
	x ^= x >> 29, x *= 0xBF58476D1CE4E5B9ULL, x ^= x >> 32, x |= 1;
	unsigned kind = ver == 255 ? 1 : (x >> 3) % 8;
	size_t q = kind == 1 ? BLK : (256 + (x >> 8) % 1792) & ~7UL;

	if (!kind) {
		memset(p, 0, BLK);
		return;
	}
	for (size_t i = 0; i < q; i += 8) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		memcpy(p + i, &x, 8);
	}
	for (size_t i = q; i < BLK; i += 8) {
		uint64_t w = blk * 31 + (i & 56);
		memcpy(p + i, &w, 8);
	}
}

static uint8_t *tver;
static char *ta, *tb;

static int put_pages(int fd, uint64_t blk, unsigned n, int bump)
{
	for (unsigned i = 0; i < n; i++)
		pattern(ta + i * BLK, blk + i, bump < 0 ? 255 : (tver[blk + i] + bump) % 200);
	int err = nbd_io(fd, NBD_CMD_WRITE, blk * BLK, ta, n * BLK);
	if (!err)
		for (unsigned i = 0; i < n; i++)
			tver[blk + i] = bump < 0 ? 255 : (tver[blk + i] + bump) % 200;
	return err;
}

static void check_pages(int fd, uint64_t blk, unsigned n, const char *when)
{
	for (unsigned i = 0; i < n; i++)
		pattern(ta + i * BLK, blk + i, tver[blk + i]);
	if (nbd_io(fd, NBD_CMD_READ, blk * BLK, tb, n * BLK) || memcmp(ta, tb, n * BLK))
		die("selftest: MISMATCH %s, near block %llu", when, (unsigned long long)blk);
}

static void check_all(int fd, const char *when)
{
	uint64_t npages = dev_size / BLK;
	for (uint64_t b = 0; b < npages; b += MAX_PAGES)
		check_pages(fd, b, npages - b < MAX_PAGES ? npages - b : MAX_PAGES, when);
}

static void selftest_compress(int threads)
{
	int fds[MAX_THREADS];
	pthread_t tw[MAX_THREADS], tc[MAX_THREADS];
	uint64_t npages = dev_size / BLK, x = 2463534242ULL;
	double t;

	tver = calloc(npages, 1);
	ta = malloc(BUFSZ), tb = malloc(BUFSZ);
	for (int i = 0; i < threads; i++) {
		int sp[2];
		if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp))
			die("socketpair: %s", strerror(errno));
		fds[i] = sp[0];
		pthread_create(&tw[i], NULL, worker, (void *)(intptr_t)sp[1]);
	}
	int fd = fds[0];

	t = now();
	for (uint64_t b = 0; b < npages; b += MAX_PAGES)
		if (put_pages(fd, b, npages - b < MAX_PAGES ? npages - b : MAX_PAGES, 0))
			die("selftest: write failed at block %llu", (unsigned long long)b);
	printf("sequential write  %7.0f MiB/s\n", dev_size / MiB / (now() - t));
	printf("stored            %7llu MiB of pages in %llu MiB (%.2fx; the device offers %.2fx)\n",
	       (unsigned long long)(dev_size / MiB), (unsigned long long)(cs.bytes / MiB),
	       (double)dev_size / cs.bytes, cratio);
	if (spill_fd >= 0)
		printf("                  %7llu MiB of that in VRAM, %llu MiB spilled to the file\n",
		       (unsigned long long)((cs.bytes - cs.sbytes) / MiB), (unsigned long long)(cs.sbytes / MiB));
	t = now();
	check_all(fd, "in sequential read");
	printf("sequential read   %7.0f MiB/s  (verified)\n", dev_size / MiB / (now() - t));

	/* Rewrite pages at random until the log has wrapped and segments have had to be cleaned. */
	uint64_t ops = npages < 400000 ? npages : 400000;
	t = now();
	for (uint64_t i = 0; i < ops; i++) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		if (put_pages(fd, x % npages, 1, 1))
			die("selftest: 4K write failed");
	}
	double dw = now() - t;
	printf("4K random write   %7.0f IOPS  (%.1f us each, 1 thread, %llu segments cleaned)\n", ops / dw,
	       dw / ops * 1e6, (unsigned long long)cs.cleaned);
	check_all(fd, "after random rewrites");
	t = now();
	for (int i = 0; i < RAND_OPS; i++) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		check_pages(fd, x % npages, 1, "in 4K random read");
	}
	double dr = now() - t;
	printf("4K random read    %7.0f IOPS  (%.1f us each, 1 thread, verified)\n", RAND_OPS / dr,
	       dr / RAND_OPS * 1e6);
	t = now();
	for (int i = 0; i < threads; i++)
		pthread_create(&tc[i], NULL, rand_reader, (void *)(intptr_t)fds[i]);
	for (int i = 0; i < threads; i++)
		pthread_join(tc[i], NULL);
	printf("4K random read    %7.0f IOPS  (%d threads)\n", (double)RAND_OPS * threads / (now() - t),
	       threads);

	/* Data that does not compress must fill the store, be refused cleanly, and harm nothing. */
	uint64_t b = 0;
	int err = 0;
	for (; b + 64 <= npages && !(err = put_pages(fd, b, 64, -1)); b += 64)
		;
	if (spill_fd < 0) {
		if (err != ENOSPC)
			die("selftest: filling with incompressible data gave error %d, not ENOSPC", err);
		check_all(fd, "after the store filled up");
		printf("incompressible    writes refused once %llu MiB of VRAM was full; contents intact\n",
		       (unsigned long long)(cs.bytes / MiB));
	} else {
		if (err && (err != ENOSPC || spill_size >= dev_size))
			die("selftest: filling with incompressible data gave error %d", err);
		t = now();
		check_all(fd, "after spilling");
		printf("incompressible    %llu MiB in VRAM, %llu MiB spilled%s; all read back at %.0f MiB/s\n",
		       (unsigned long long)((cs.bytes - cs.sbytes) / MiB), (unsigned long long)(cs.sbytes / MiB),
		       err ? ", then refused" : "", dev_size / MiB / (now() - t));
		/* Free the newest half, which is what is in VRAM: the spilled pages should come back. */
		uint64_t half = npages / 2 & ~63ULL;
		for (uint64_t off = half * BLK; off < dev_size; off += 1024 * MiB) {
			uint64_t n = dev_size - off < 1024 * MiB ? dev_size - off : 1024 * MiB;
			if (nbd_io(fd, NBD_CMD_TRIM, off, NULL, n))
				die("selftest: discard failed");
		}
		t = now();
		cs.returned = 0;
		while (cs.sbytes && (cs.bytes - cs.sbytes) * 100 < phys_size * low_pct && now() - t < 120)
			usleep(20000);
		if (cs.sbytes && (cs.bytes - cs.sbytes) * 100 < phys_size * low_pct)
			die("selftest: spilled pages did not come back to a card with room for them");
		for (uint64_t c = 0; c < half; c += MAX_PAGES)
			check_pages(fd, c, half - c < MAX_PAGES ? half - c : MAX_PAGES, "after pages came back");
		printf("rehydration       %llu MiB came back from the file in %.1f s, %llu MiB left there; verified\n",
		       (unsigned long long)(cs.returned / MiB), now() - t,
		       (unsigned long long)(cs.sbytes / MiB));
	}

	for (uint64_t off = 0; off < dev_size; off += 1024 * MiB) {
		uint64_t n = dev_size - off < 1024 * MiB ? dev_size - off : 1024 * MiB;
		if (nbd_io(fd, NBD_CMD_TRIM, off, NULL, n))
			die("selftest: discard failed");
	}
	memset(tver, 0, npages);
	memset(ta, 0, BLK);
	if (cs.pages || cs.bytes || nbd_io(fd, NBD_CMD_READ, (npages / 2) * BLK, tb, BLK) || memcmp(ta, tb, BLK))
		die("selftest: discard did not empty the store");
	if (put_pages(fd, 7, 3, 1))
		die("selftest: write after discard failed");
	check_pages(fd, 7, 3, "after discard");
	printf("discard               ok\n");

	for (int i = 0; i < threads; i++) {
		struct nbd_req rq = { htobe32(NBD_REQUEST_MAGIC), htobe32(NBD_CMD_DISC), 0, 0, 0 };
		wr_full(fds[i], &rq, sizeof rq, 0);
		pthread_join(tw[i], NULL);
	}
	printf("selftest passed\n");
}

int main(int argc, char **argv)
{
	uint64_t size_mib = 0;
	int gpu = 0, threads = 0, test = 0, do_bench = 0, ublk;

	for (int i = 1; i < argc; i++) {
		char *v = i + 1 < argc ? argv[i + 1] : NULL;
		if (!strcmp(argv[i], "--selftest"))
			test = 1;
		else if ((!strcmp(argv[i], "--dev") || !strcmp(argv[i], "--nbd")) && v)
			dev_path = v, i++;
		else if (!strcmp(argv[i], "--bench") && v)
			dev_path = v, do_bench = 1, i++;
		else if (!strcmp(argv[i], "--size") && v)
			size_mib = strtoull(v, NULL, 10), i++;
		else if (!strcmp(argv[i], "--gpu") && v)
			gpu = atoi(v), i++;
		else if (!strcmp(argv[i], "--threads") && v)
			threads = atoi(v), i++;
		else if (!strcmp(argv[i], "--compress") && v)
			cratio = atof(v), i++;
		else if (!strcmp(argv[i], "--zlevel") && v)
			clevel = atoi(v), i++;
		else if (!strcmp(argv[i], "--spill") && v)
			spill_path = *v ? v : NULL, i++;
		else if (!strcmp(argv[i], "--spill-size") && v)
			spill_size = strtoull(v, NULL, 10) * MiB, i++;
		else if (!strcmp(argv[i], "--high") && v)
			high_pct = atoi(v), i++;
		else if (!strcmp(argv[i], "--low") && v)
			low_pct = atoi(v), i++;
		else
			die("usage: vroom (--selftest | --dev /dev/nbdN | --dev /dev/ublkbN | --bench DEV)"
			    " [--size MiB] [--gpu N] [--threads N]"
			    " [--compress RATIO [--spill FILE [--spill-size MiB]]]");
	}
	if (cratio && (cratio < 1 || cratio > 8))
		die("--compress takes how much larger than the VRAM to make the device, 1 to 8 (0: off)");
	if (spill_path && !cratio)
		die("--spill needs --compress");
	if (low_pct < 1 || high_pct > 95 || low_pct + 5 > high_pct)
		die("--low and --high are percentages of the VRAM, at least 5 apart, at most 95");
	if (test == !!dev_path)
		die("give exactly one of --selftest, --dev DEVICE or --bench DEVICE");
	/*
	 * CUDA serialises copies within a context. NBD still gains from several
	 * threads (the socket work runs in parallel); ublk is fastest with one queue.
	 */
	ublk = !test && !strncmp(dev_path, "/dev/ublkb", 10);
	if (!threads)
		threads = ublk && !do_bench ? 1 : 4;
	if (threads < 1 || threads > MAX_THREADS)
		die("--threads must be 1..%d", MAX_THREADS);
	if (test && !size_mib)
		size_mib = 1024;

	setvbuf(stdout, NULL, _IOLBF, 0);
	if (do_bench) {
		bench(threads);
		return 0;
	}
	/*
	 * Block stop signals before CUDA starts its helper threads: every thread
	 * must inherit the mask, or SIGTERM would be delivered to one of them
	 * and kill us while swap is live.
	 */
	if (!test)
		block_stop_signals();
	vram_init(gpu, size_mib);
	if (getenv("VROOM_COUNT") && atoi(getenv("VROOM_COUNT"))) {
		cn_init();
		counting = 1;
	}
	if (test && cratio)
		selftest_compress(threads);
	else if (test)
		selftest(threads);
	else if (ublk)
		serve_ublk(threads);
	else
		serve_nbd(threads);
	if (spill_path)
		unlink(spill_path);
	if (counting && test)
		count_print(stdout);
	return 0;
}
