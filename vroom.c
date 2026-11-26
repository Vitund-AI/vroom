/*
 * SPDX-License-Identifier: MIT
 * vroom - expose idle NVIDIA VRAM as a block device (for swap).
 *
 * VRAM is allocated through the driver's own CUDA library (libcuda.so.1,
 * shipped with the NVIDIA driver; no CUDA toolkit needed) and served to the
 * kernel as /dev/ublkbN through io_uring, or as /dev/nbdN over in-process
 * socketpairs speaking the NBD transmission protocol. Every block I/O is one
 * host<->device copy across PCIe.
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
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/eventfd.h>
#include <sys/prctl.h>
#include <poll.h>
#include <sys/random.h>
#include <sys/resource.h>
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

/*
 * Something has gone wrong while serving. Say so, but not without limit: what
 * reads our log may itself be waiting for a page from us, and a thread that
 * blocks writing to it holds up everyone behind it.
 */
static int say(void)
{
	static uint64_t said;
	uint64_t n = __atomic_add_fetch(&said, 1, __ATOMIC_RELAXED);
	return n <= 20 || n % 1000 == 0;
}

static void spawn(void *(*fn)(void *), void *arg, pthread_t *out)
{
	pthread_t t;
	int r = pthread_create(out ? out : &t, NULL, fn, arg);
	if (r)
		die("cannot start a thread: %s", strerror(r));
}

static void *sym(void *lib, const char *name)
{
	void *p = dlsym(lib, name);
	if (!p)
		die("libcuda.so.1 lacks %s", name);
	return p;
}

#ifdef FAKE_CARD
/*
 * A card made of ordinary memory, for 'make check': everything above the
 * eleven driver calls can then be tested, under the sanitizers, on a machine
 * with no GPU. Never part of the installed program.
 */
static int fk_init(unsigned f) { (void)f; return 0; }
static int fk_dev(int *d, int i) { *d = i; return 0; }
static int fk_name(char *n, int len, int d) { (void)d; snprintf(n, len, "fake card"); return 0; }
static int fk_ctx(CUcontext *c, unsigned f, int d) { (void)f, (void)d; *c = NULL; return 0; }
static int fk_cur(CUcontext c) { (void)c; return 0; }
static int fk_info(size_t *f, size_t *t) { *f = *t = (size_t)4096 * MiB; return 0; }
static int fk_alloc(CUdeviceptr *p, size_t n) { void *m = malloc(n); *p = (uintptr_t)m; return m ? 0 : 2; }
static int fk_host(void **p, size_t n) { return posix_memalign(p, BLK, n) ? 2 : 0; }
static int fk_set(CUdeviceptr p, unsigned char v, size_t n) { memset((void *)(uintptr_t)p, v, n); return 0; }
static int fk_in(CUdeviceptr d, const void *s, size_t n) { memcpy((void *)(uintptr_t)d, s, n); return 0; }
static int fk_out(void *d, CUdeviceptr s, size_t n) { memcpy(d, (void *)(uintptr_t)s, n); return 0; }
#endif

static void vram_init(int gpu, uint64_t size_mib)
{
#ifdef FAKE_CARD
	cu.Init = fk_init, cu.DeviceGet = fk_dev, cu.DeviceGetName = fk_name, cu.CtxCreate = fk_ctx;
	cu.CtxSetCurrent = fk_cur, cu.MemGetInfo = fk_info, cu.MemAlloc = fk_alloc, cu.MemAllocHost = fk_host;
	cu.MemsetD8 = fk_set, cu.HtoD = fk_in, cu.DtoH = fk_out;
#else
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
#endif

	int dev, r;
	char name[128] = "?";
	size_t free_b, total_b;
	if ((r = cu.Init(0)))
		die("cuInit failed (CUDA error %d)%s", r,
		    r == 803 ? ": the NVIDIA driver has been upgraded and its kernel module has not; reboot" : "");
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
	if (!chunks)
		die("out of memory");
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

/*
 * How fast the card and the spill file are being, for 'vroomctl stats'. One
 * copy in SAMPLE is timed: reading the clock costs about a hundredth of the
 * quickest copy, so even that much is caution. Everything is counted.
 */
#define SAMPLE 16
#define HIST 112 /* buckets a quarter of a doubling wide, from 64 ns to 4 minutes */
enum { P_REQ_READ, P_REQ_WRITE, P_VRAM_READ, P_VRAM_WRITE, P_DISK_READ, P_DISK_WRITE, P_KINDS };
static struct perf {
	uint64_t ops, bytes;		 /* all of them */
	uint64_t t_ops, t_bytes, t_ns;	 /* the ones that were timed */
	uint64_t hist[HIST];		 /* how long those took */
} perf[P_KINDS];

static unsigned hist_bucket(uint64_t ns)
{
	if (ns < 64)
		return 0;
	unsigned b = 63 - __builtin_clzll(ns), i = (b - 6) * 4 + ((ns >> (b - 2)) & 3);
	return i < HIST ? i : HIST - 1;
}

/* A time in nanoseconds from the middle of bucket i. */
static double hist_ns(unsigned i)
{
	return (double)((uint64_t)(4 | (i & 3)) << (i / 4 + 4)) * 1.09;
}

static uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Count an operation; returns the time if this one is to be timed, else 0. */
static uint64_t perf_begin(int kind, size_t len)
{
	__atomic_add_fetch(&perf[kind].bytes, len, __ATOMIC_RELAXED);
	return __atomic_fetch_add(&perf[kind].ops, 1, __ATOMIC_RELAXED) % SAMPLE ? 0 : now_ns();
}

static void perf_end(int kind, size_t len, uint64_t t0)
{
	if (!t0)
		return;
	uint64_t ns = now_ns() - t0;
	__atomic_add_fetch(&perf[kind].hist[hist_bucket(ns)], 1, __ATOMIC_RELAXED);
	__atomic_add_fetch(&perf[kind].t_ns, ns, __ATOMIC_RELAXED);
	__atomic_add_fetch(&perf[kind].t_bytes, len, __ATOMIC_RELAXED);
	__atomic_add_fetch(&perf[kind].t_ops, 1, __ATOMIC_RELAXED);
}

static int vram_rw(int wr, uint64_t off, char *buf, size_t len)
{
	size_t whole = len;
	uint64_t t0 = perf_begin(wr ? P_VRAM_WRITE : P_VRAM_READ, len);

	while (len) {
		uint64_t ci = off / CHUNK, co = off % CHUNK;
		size_t n = len < CHUNK - co ? len : CHUNK - co;
		int r = wr ? cu.HtoD(chunks[ci] + co, buf, n) : cu.DtoH(buf, chunks[ci] + co, n);
		if (r) {
			if (say())
				fprintf(stderr, "vroom: VRAM %s failed at offset %llu (CUDA error %d)\n",
				wr ? "write" : "read", (unsigned long long)off, r);
			return -1;
		}
		off += n, buf += n, len -= n;
	}
	perf_end(wr ? P_VRAM_WRITE : P_VRAM_READ, whole, t0);
	return 0;
}

/*
 * ---- compressed store (--compress RATIO) ----
 *
 * Each 4K page is compressed by itself into a record and appended, behind an
 * 8-byte header naming the record, to a log laid over the VRAM in 1 MiB
 * segments. Tables in RAM say which record each page points at and where
 * each record is. Pages with identical contents point at the same record: a
 * page arriving is fingerprinted, and if a record has that fingerprint the
 * page just points at it, wherever the record now is. A record nothing
 * points at any more is dead; a segment with nothing live in it is free
 * again, and when free segments run short the one with the least live data
 * is read back and its survivors appended afresh.
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
#define REC_HDR 32
#define RAW_ABOVE 3584 /* a page that does not get below this is stored as it is */
#define STAGE_BYTES (BUFSZ + BUFSZ / BLK * REC_HDR + BLK)
/* Reads from the spill file are block-aligned, so each run of records can carry two blocks of slack. */
#define STAGE_SPILL ((size_t)MAX_PAGES * 3 * BLK + BLK)
#define MAX_PAGES (BUFSZ / BLK)
#define E_VALID (1ULL << 63)
#define E_OFF(e) ((e) & ((1ULL << 40) - 1))
#define E_LEN(e) ((uint32_t)((e) >> 40) & 0x1fff)
#define E_MAKE(off, len) (E_VALID | (uint64_t)(len) << 40 | (off))
#define IDX_GONE UINT32_MAX
/* SEG_BAD: its record headers no longer add up; what is in it can be read but is never moved. */
enum { SEG_FREE, SEG_OPEN, SEG_CLOSED, SEG_CLEANING, SEG_BAD };

struct rec {
	uint32_t id; /* which record this is: see cs.rloc */
	uint16_t len, pad;
	uint64_t nonce;	 /* never used twice in the life of the key */
	uint8_t tag[16]; /* proves the data, the id and the length are what was stored */
};

/*
 * Everything stored is encrypted, record by record, under a key made up at
 * start and never written anywhere: what is left on the card or in the spill
 * file once the daemon has gone is noise. The cipher also authenticates, so a
 * record that has been altered or has rotted is an I/O error, not wrong data.
 * AES-256-GCM unless --cipher chacha20 asks for ChaCha20-Poly1305, which is
 * the one to use on a CPU with no AES instructions. From libcrypto.so.3.
 */
static struct {
	void *(*CtxNew)(void);
	const void *(*Aes)(void);
	const void *(*ChaCha)(void);
	int (*Init)(void *, const void *, void *, const unsigned char *, const unsigned char *, int);
	int (*Update)(void *, unsigned char *, int *, const unsigned char *, int);
	int (*Final)(void *, unsigned char *, int *);
	int (*Ctrl)(void *, int, int, void *);
} ev;
static const char *cipher_name = "aes";
static const void *cipher;
static unsigned char cipher_key[32], print_key[32];
static uint64_t next_nonce;

/*
 * A page's fingerprint: the cipher's own 128-bit authenticator run over the
 * page under a second key that never leaves this process. Two pages with the
 * same fingerprint are taken to be the same page, without reading the stored
 * one back, which is what lets a page be matched against a copy that has
 * gone to the spill file. Someone who does not know the key cannot make two
 * different pages collide, and the chance of it happening by itself is about
 * 1 in 2^100 for every pair of pages.
 */
struct fp {
	uint64_t a, b;
};

static struct {
	void *(*CreateCCtx)(void);
	void *(*CreateDCtx)(void);
	size_t (*Compress)(void *, void *, size_t, const void *, size_t, int);
	size_t (*Decompress)(void *, void *, size_t, const void *, size_t);
	unsigned (*IsError)(size_t);
} z;

static int clevel = 1;
static uint64_t hash_secret;
static struct {
	pthread_mutex_t lock;
	uint32_t *map;	 /* per page: the record with its contents, or 0 if it reads as zeros */
	/* Per record. Pages with identical contents share one, so it can move without touching them. */
	uint64_t *rloc;	 /* where it is: E_MAKE(offset, length); 0 until it has been written */
	struct fp *rfp;	 /* fingerprint of the page it holds */
	uint32_t *rnon;	 /* the low half of the nonce it was sealed with: see cs_read */
	uint32_t *rref;	 /* how many pages point at it; for an unused id, the next unused one */
	uint32_t rfree, rmax;
	uint32_t *idx;	 /* open-addressed: hash -> a record holding that page */
	uint64_t idx_mask, idx_used;
	/* Per segment */
	uint32_t *live;	 /* bytes of records still current */
	uint32_t *used;	 /* bytes appended so far */
	uint8_t *state;
	uint64_t *seq;	 /* when it was opened, for finding the oldest */
	uint16_t *pins;	 /* readers at work in a spill segment, which must not be rewritten meanwhile */
	uint8_t *hits;	 /* reads in a row that each went a little further into a spill segment */
	uint32_t *lastoff; /* where in the segment the last of them ended */
	/*
	 * Spill segments being read through in order are read once, whole, and
	 * kept here: a megabyte in one read costs about three single pages.
	 */
	struct held {
		uint32_t seg;
		uint64_t seq, used; /* which writing of that segment; when last used */
		int valid, loading;
		char *buf;
	} held[16];
	uint64_t held_clock, held_hits;
	pthread_cond_t wake, room; /* work for the keeper; room made for writers */
	int stuck;		   /* the keeper can free nothing more: the store is full */
	int waiting;		   /* writes waiting for room */
	uint32_t ahead;		   /* free segments the keeper tries to keep in hand */
	uint64_t waits;		   /* writes that had to wait for it */
	uint32_t *freelist, nseg, nfree, open;
	uint32_t *sfree, nspill, nsfree; /* spill segments are numbered nseg onwards */
	char *gcbuf, *evbuf, *rebuf;
	uint64_t clock;
	uint64_t pages, recs, bytes, sbytes; /* pages held, the records behind them, their bytes, and how much of that is spilled */
	uint64_t cleaned, refused, evicted, returned;
	uint64_t reread, mended, lost; /* pages that failed their check: right at a second reading, put right, or gone */
} cs = { .lock = PTHREAD_MUTEX_INITIALIZER, .wake = PTHREAD_COND_INITIALIZER,
	 .room = PTHREAD_COND_INITIALIZER };

static const char *spill_path;
static int spill_fd = -1, spill_is_dev;
static uint64_t spill_size, spill_base; /* a disk starts with a label, and the segments follow it */
static const char spill_magic[16] = "vroom spill v1\n";
static int high_pct = 90, low_pct = 70;
static void *keeper_thread(void *arg);

/* Per-thread compressor state and page-locked buffer for records on their way in or out. */
static __thread struct {
	void *cc, *dc, *seal, *open, *print;
	char *stage;
	char *fix, *work; /* for rescue(): a record as it was read, and a copy to try things on */
} ct;

static int page_print(const char *page, struct fp *f);
static int rec_crypt(int seal, struct rec *r, char *data);

static void ct_init(void)
{
	void *p;
	if (ct.stage)
		return;
	size_t sz = spill_fd >= 0 ? STAGE_SPILL : STAGE_BYTES;
	if (cu.MemAllocHost(&p, sz) && posix_memalign(&p, BLK, sz))
		die("cannot allocate staging buffer");
	ct.stage = p;
	if (posix_memalign(&p, BLK, 6 * BLK))
		die("cannot allocate staging buffer");
	ct.fix = memset(p, 0, 6 * BLK), ct.work = ct.fix + 3 * BLK;
	ct.cc = z.CreateCCtx();
	ct.dc = z.CreateDCtx();
	ct.seal = ev.CtxNew();
	ct.open = ev.CtxNew();
	ct.print = ev.CtxNew();
	if (!ct.cc || !ct.dc || !ct.seal || !ct.open || !ct.print ||
	    ev.Init(ct.seal, cipher, NULL, cipher_key, NULL, 1) != 1 ||
	    ev.Init(ct.open, cipher, NULL, cipher_key, NULL, 0) != 1 ||
	    ev.Init(ct.print, cipher, NULL, print_key, NULL, 1) != 1)
		die("cannot set up compression and encryption");
	/*
	 * Use all of it once now. The libraries allocate their working memory
	 * the first time they are used, and that must not be when the machine
	 * is already out of memory and waiting on us.
	 */
	struct rec r = { .id = 0 };
	struct fp f;
	char *page = ct.stage, *packed = ct.stage + BLK, *back = ct.stage + 2 * BLK;
	for (size_t i = 0; i < BLK; i++)
		page[i] = i / 64;
	size_t c = z.Compress(ct.cc, packed, RAW_ABOVE, page, BLK, clevel);
	if (z.IsError(c) || !c || c > RAW_ABOVE)
		die("cannot compress");
	r.len = c;
	if (page_print(page, &f) || rec_crypt(1, &r, packed) || rec_crypt(0, &r, packed) ||
	    z.Decompress(ct.dc, back, BLK, packed, c) != BLK || memcmp(page, back, BLK))
		die("compression and encryption do not give back what went in");
}

/* Threads that serve requests get ready before the device exists; the starter waits for them here. */
static int threads_ready;
static void thread_ready(void)
{
	if (cratio)
		ct_init();
	__atomic_add_fetch(&threads_ready, 1, __ATOMIC_RELEASE);
}

static void await_threads(int n)
{
	while (__atomic_load_n(&threads_ready, __ATOMIC_ACQUIRE) < n)
		usleep(1000);
}

static int page_print(const char *page, struct fp *f)
{
	static const unsigned char iv[12];
	unsigned char end[16];
	int n;

	return ev.Init(ct.print, NULL, NULL, NULL, iv, -1) != 1 ||
	       ev.Update(ct.print, NULL, &n, (const unsigned char *)page, BLK) != 1 ||
	       ev.Final(ct.print, end, &n) != 1 || ev.Ctrl(ct.print, 0x10 /* get tag */, 16, f) != 1;
}

/* Encrypt (seal) or check and decrypt a record's data where it lies. r->id and r->len must be set. */
static int rec_crypt(int seal, struct rec *r, char *data)
{
	void *c = seal ? ct.seal : ct.open;
	unsigned char iv[12] = { 0 }, *d = (unsigned char *)data;
	int n;

	if (seal)
		r->nonce = __atomic_add_fetch(&next_nonce, 1, __ATOMIC_RELAXED);
	memcpy(iv, &r->nonce, 8);
	if (ev.Init(c, NULL, NULL, NULL, iv, -1) != 1 ||
	    ev.Update(c, NULL, &n, (unsigned char *)r, 8) != 1 || /* the id and length are vouched for too */
	    ev.Update(c, d, &n, d, r->len) != 1)
		return -1;
	if (!seal && ev.Ctrl(c, 0x11 /* set tag */, 16, r->tag) != 1)
		return -1;
	if (ev.Final(c, d + n, &n) != 1)
		return -1;
	return seal && ev.Ctrl(c, 0x10 /* get tag */, 16, r->tag) != 1 ? -1 : 0;
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
	if (!(lib = dlopen("libcrypto.so.3", RTLD_NOW)))
		die("--compress needs libcrypto.so.3: %s", dlerror());
	ev.CtxNew = sym(lib, "EVP_CIPHER_CTX_new");
	ev.Aes = sym(lib, "EVP_aes_256_gcm");
	ev.ChaCha = sym(lib, "EVP_chacha20_poly1305");
	ev.Init = sym(lib, "EVP_CipherInit_ex");
	ev.Update = sym(lib, "EVP_CipherUpdate");
	ev.Final = sym(lib, "EVP_CipherFinal_ex");
	ev.Ctrl = sym(lib, "EVP_CIPHER_CTX_ctrl");
	if (!strcmp(cipher_name, "aes"))
		cipher = ev.Aes();
	else if (!strcmp(cipher_name, "chacha20"))
		cipher = ev.ChaCha();
	else
		die("--cipher is aes or chacha20");
	if (getrandom(cipher_key, sizeof cipher_key, 0) != sizeof cipher_key ||
	    getrandom(print_key, sizeof print_key, 0) != sizeof print_key)
		die("cannot get random bytes for the keys: %s", strerror(errno));
	/* Nobody outside should be able to make two pages hash alike on purpose. */
	if (getrandom(&hash_secret, sizeof hash_secret, 0) != sizeof hash_secret)
		hash_secret = (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ULL ^ (uintptr_t)&lib;

	cs.nseg = phys_size / SEG;
	if (spill_path) {
		/*
		 * The spill can be a file, which is made here and removed at the end,
		 * or a whole disk or partition, which is simply used: everything on
		 * it is overwritten. O_EXCL makes the kernel refuse one that is
		 * mounted, or has a partition mounted, or is otherwise in use.
		 * O_DIRECT: nothing of ours should sit in the page cache while memory
		 * is short.
		 */
		struct stat st;
		spill_is_dev = !stat(spill_path, &st) && S_ISBLK(st.st_mode);
		if (spill_is_dev) {
			uint64_t whole = 0, range[2] = { 0, 0 };
			spill_fd = open(spill_path, O_RDWR | O_CLOEXEC | O_DIRECT | O_EXCL);
			if (spill_fd < 0)
				die("--spill %s: %s%s", spill_path, strerror(errno),
				    errno == EBUSY ? " (it, or a partition on it, is in use)" : "");
			if (ioctl(spill_fd, BLKGETSIZE64, &whole) || whole < 16 * MiB)
				die("--spill %s: cannot tell its size", spill_path);
			/*
			 * The first megabyte is a label saying the disk is ours, and
			 * otherwise zeros, so that nothing takes what follows for a
			 * filesystem. vroomctl looks for the label before every start:
			 * a disk without one has to be empty, which is what stops a
			 * name such as /dev/sdb that now means a different disk.
			 */
			spill_base = SEG;
			whole -= spill_base;
			if (!spill_size || spill_size > whole)
				spill_size = dev_size < whole ? dev_size : whole;
			spill_size &= ~(uint64_t)(SEG - 1);
			range[1] = spill_base + spill_size; /* tell an SSD that all of this is free to start with */
			ioctl(spill_fd, BLKDISCARD, range);
			if (posix_memalign(&p, BLK, SEG))
				die("out of memory");
			memset(p, 0, SEG);
			memcpy(p, spill_magic, sizeof spill_magic);
			if (pwrite(spill_fd, p, SEG, 0) != SEG)
				die("--spill %s: cannot write to it: %s", spill_path, strerror(errno));
			free(p);
		} else {
			/*
			 * A file of our own making, never one that is already there,
			 * and with no name once it is open: it goes when we do, however
			 * that happens. Without O_DIRECT every page moved out would
			 * first take up memory in the page cache, which is the one
			 * thing in short supply.
			 */
			spill_fd = open(spill_path, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_DIRECT, 0600);
			if (spill_fd < 0)
				die("--spill %s: %s%s", spill_path, strerror(errno),
				    errno == EEXIST ? " (if it is left over from an earlier run, remove it)" :
				    errno == EINVAL ? " (the filesystem cannot do direct I/O; put it on a disk)" : "");
			unlink(spill_path);
			if (!spill_size)
				spill_size = dev_size;
			int r = posix_fallocate(spill_fd, 0, spill_size);
			if (r)
				die("--spill %s: cannot reserve %llu MiB: %s", spill_path,
				    (unsigned long long)(spill_size / MiB), strerror(r));
		}
		if (phys_size + spill_size >= 1ULL << 40)
			die("VRAM and spill together must come to less than 1024 GiB");
		cs.nspill = spill_size / SEG;
		fprintf(stderr, "vroom: spilling to %s (%llu MiB) above %d%% full, back below %d%%\n",
			spill_path, (unsigned long long)(spill_size / MiB), high_pct, low_pct);
	}
	uint32_t all = cs.nseg + cs.nspill;
	uint64_t npages = dev_size / BLK, cap = 1;
	/* a request's new records exist briefly alongside the ones they replace, in every thread at once */
	cs.rmax = npages + MAX_THREADS * MAX_PAGES;
	while (cap < 2ULL * cs.rmax)
		cap <<= 1;
	cs.idx_mask = cap - 1;
	cs.map = calloc(npages, sizeof *cs.map);
	cs.rloc = calloc(cs.rmax + 1, sizeof *cs.rloc);
	cs.rfp = calloc(cs.rmax + 1, sizeof *cs.rfp);
	cs.rref = calloc(cs.rmax + 1, sizeof *cs.rref);
	cs.rnon = calloc(cs.rmax + 1, sizeof *cs.rnon);
	cs.idx = calloc(cap, sizeof *cs.idx);
	cs.live = calloc(all, sizeof *cs.live);
	cs.used = calloc(all, sizeof *cs.used);
	cs.seq = calloc(all, sizeof *cs.seq);
	cs.pins = calloc(all, sizeof *cs.pins);
	cs.hits = calloc(all, 1);
	cs.lastoff = calloc(all, sizeof *cs.lastoff);
	for (int i = 0; spill_path && i < 16; i++)
		if (posix_memalign(&p, BLK, SEG))
			die("out of memory for the spill read-ahead");
		else
			cs.held[i].buf = p;
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
	if (!cs.map || !cs.rloc || !cs.rfp || !cs.rref || !cs.rnon || !cs.idx || !cs.live || !cs.used || !cs.seq ||
	    !cs.pins || !cs.hits || !cs.lastoff || !cs.state || !cs.freelist || !cs.sfree || !cs.gcbuf || !cs.evbuf || !cs.rebuf)
		die("out of memory for the compressed store's tables");
	for (uint32_t id = 1; id <= cs.rmax; id++)
		cs.rref[id] = id < cs.rmax ? id + 1 : 0;
	cs.rfree = 1;
	for (uint32_t s = cs.nseg; s-- > 1;)
		cs.freelist[cs.nfree++] = s;
	for (uint32_t s = cs.nspill; s-- > 0;)
		cs.sfree[cs.nsfree++] = cs.nseg + s;
	cs.open = 0;
	cs.state[0] = SEG_OPEN;
	cs.seq[0] = ++cs.clock;
	cs.ahead = cs.nseg / 32 < 4 ? 4 : cs.nseg / 32 > 64 ? 64 : cs.nseg / 32;
	spawn(keeper_thread, NULL, NULL);
}

static uint64_t page_hash(const char *p)
{
	uint64_t h[4] = { 0x9E3779B97F4A7C15ULL ^ hash_secret, 0xBF58476D1CE4E5B9ULL + hash_secret,
			  0x94D049BB133111EBULL ^ hash_secret >> 7, 0xD6E8FEB86659FD93ULL ^ hash_secret << 9 };
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

static void seg_free(uint32_t s)
{
	cs.state[s] = SEG_FREE;
	cs.used[s] = 0;
	cs.hits[s] = 0;
	cs.stuck = 0; /* space has come back: worth trying again */
	if (s < cs.nseg)
		cs.freelist[cs.nfree++] = s;
	else
		cs.sfree[cs.nsfree++] = s;
}

/* What we hold of a spill segment is no longer what the file holds. */
static void held_drop(uint32_t s)
{
	for (int i = 0; i < 16; i++)
		if (cs.held[i].seg == s)
			cs.held[i].valid = 0, cs.held[i].seq = 0;
}

/* A record's bytes arrive at (add) or leave the place e describes. */
static void loc_account(uint64_t e, int add)
{
	uint32_t s = E_OFF(e) / SEG, sz = REC_HDR + E_LEN(e);
	if (add) {
		cs.bytes += sz;
		cs.live[s] += sz;
		if (s >= cs.nseg)
			cs.sbytes += sz;
		return;
	}
	cs.bytes -= sz;
	cs.live[s] -= sz;
	if (s >= cs.nseg)
		cs.sbytes -= sz;
	cs.stuck = 0; /* something has died: there may be room to be made again */
	if (!cs.live[s] && (cs.state[s] == SEG_CLOSED || cs.state[s] == SEG_BAD))
		seg_free(s);
}

/* A record has been written at e, for the first time or because it was moved. */
static void rec_place(uint32_t id, uint64_t e)
{
	uint64_t old = cs.rloc[id];
	cs.rloc[id] = e;
	loc_account(e, 1);
	if (old)
		loc_account(old, 0);
}

static void idx_insert(uint32_t id)
{
	uint64_t i = cs.rfp[id].a & cs.idx_mask;
	while (cs.idx[i] && cs.idx[i] != IDX_GONE)
		i = (i + 1) & cs.idx_mask;
	cs.idx_used += !cs.idx[i];
	cs.idx[i] = id;
}

/*
 * Make sure n more records can go into the index. When it is mostly the slots
 * of records long gone it is started again, from the records that are written
 * somewhere, which are exactly the ones in it. So this must be called before
 * the new ones are written, not between writing them and putting them in:
 * they would then go in twice, and a record that is in twice is still found,
 * by its old contents, after it has been freed.
 */
static void idx_reserve(size_t n)
{
	if ((cs.idx_used + n) * 10 <= cs.idx_mask * 7)
		return;
	memset(cs.idx, 0, (cs.idx_mask + 1) * sizeof *cs.idx);
	cs.idx_used = 0;
	for (uint32_t r = 1; r <= cs.rmax; r++)
		if (cs.rloc[r])
			idx_insert(r);
}

static void idx_remove(uint32_t id)
{
	for (uint64_t i = cs.rfp[id].a & cs.idx_mask; cs.idx[i]; i = (i + 1) & cs.idx_mask)
		if (cs.idx[i] == id) {
			cs.idx[i] = IDX_GONE;
			return;
		}
}

static uint32_t idx_find(const struct fp *f)
{
	for (uint64_t i = f->a & cs.idx_mask; cs.idx[i]; i = (i + 1) & cs.idx_mask)
		if (cs.idx[i] != IDX_GONE && cs.rloc[cs.idx[i]] && cs.rfp[cs.idx[i]].a == f->a &&
		    cs.rfp[cs.idx[i]].b == f->b)
			return cs.idx[i];
	return 0;
}

/* A new record with one page pointing at it, not yet written anywhere. */
static uint32_t rec_alloc(const struct fp *h)
{
	uint32_t id = cs.rfree;
	if (!id)
		return 0;
	cs.rfree = cs.rref[id];
	cs.rref[id] = 1;
	cs.rfp[id] = *h;
	cs.rloc[id] = 0;
	cs.recs++;
	return id;
}

static void rec_unref(uint32_t id)
{
	if (--cs.rref[id])
		return;
	if (cs.rloc[id]) {
		idx_remove(id);
		loc_account(cs.rloc[id], 0);
		cs.rloc[id] = 0;
	}
	cs.rfp[id] = (struct fp){ 0, 0 };
	cs.rref[id] = cs.rfree;
	cs.rfree = id;
	cs.recs--;
}

/* The page reads as zeros from now on. */
static void map_drop(uint64_t lpn)
{
	uint32_t id = cs.map[lpn];
	if (!id)
		return;
	cs.map[lpn] = 0;
	cs.pages--;
	rec_unref(id);
}

/* Point a page at a record; the caller has already counted the reference. */
static void map_set(uint64_t lpn, uint32_t id)
{
	map_drop(lpn);
	cs.map[lpn] = id;
	cs.pages++;
}

/* Append records (header, data, header, data...) to the log and note where each now is. */
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
			rec_place(r.id, E_MAKE(at + pos, r.len));
			cs.rnon[r.id] = (uint32_t)r.nonce;
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
static uint64_t moved_from[SEG / (REC_HDR + 1) + 1]; /* by evict: where each record it is moving was */
static size_t nmoved;

static size_t survivors(const char *src, uint32_t v, char *dst, size_t out, size_t limit, int *all)
{
	struct rec r;

	if (all)
		*all = 1;
	for (size_t pos = 0; pos < cs.used[v]; pos += REC_HDR + r.len) {
		int ours;
		memcpy(&r, src + pos, sizeof r);
		ours = r.id && r.id <= cs.rmax && cs.rloc[r.id] && E_OFF(cs.rloc[r.id]) == (uint64_t)v * SEG + pos;
		/*
		 * The headers come from the card or the disk, where a bit can
		 * flip, and one wrong length would send us through the rest of
		 * the segment out of step. A segment like that is left where it
		 * is: its records can still be read, each from where the tables
		 * say it is, and nothing is lost that was not lost already.
		 */
		if (r.len > BLK || !r.len || pos + REC_HDR + r.len > cs.used[v] ||
		    (ours && (r.len != E_LEN(cs.rloc[r.id]) || (uint32_t)r.nonce != cs.rnon[r.id]))) {
			if (say())
				fprintf(stderr, "vroom: segment %u is damaged at %zu; leaving it where it is\n", v, pos);
			cs.state[v] = SEG_BAD;
			if (all)
				*all = 1; /* nothing more to be had from it: go on to the next */
			break;
		}
		if (!ours)
			continue;
		if (out + REC_HDR + r.len > limit) {
			if (all)
				*all = 0;
			break;
		}
		memmove(dst + out, src + pos, REC_HDR + r.len);
		out += REC_HDR + r.len;
		if (dst == cs.evbuf)
			moved_from[nmoved++] = (uint64_t)v * SEG + pos;
	}
	return out;
}

static int spill_broken; /* the self-test's way of having the disk fail */

static int spill_io(int wr, char *buf, size_t len, uint64_t off)
{
	size_t whole = len;
	uint64_t t0 = perf_begin(wr ? P_DISK_WRITE : P_DISK_READ, len);

	if (spill_broken)
		return -1;

	while (len) {
		ssize_t n = wr ? pwrite(spill_fd, buf, len, spill_base + off) :
				 pread(spill_fd, buf, len, spill_base + off);
		if (n <= 0) {
			if (n < 0 && errno == EINTR)
				continue;
			if (say())
				fprintf(stderr, "vroom: spill %s failed: %s\n", wr ? "write" : "read",
					n ? strerror(errno) : "short");
			return -1;
		}
		buf += n, len -= n, off += n;
	}
	perf_end(wr ? P_DISK_WRITE : P_DISK_READ, whole, t0);
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
	int reuse = !cs.nsfree, all, err = 0;
	struct rec r;

	if (spill_fd < 0)
		return ENOSPC;
	nmoved = 0;
	if (reuse) {
		slot = 0;
		for (uint32_t s = cs.nseg; s < cs.nseg + cs.nspill; s++)
			if (cs.state[s] == SEG_CLOSED && cs.live[s] < best && !cs.pins[s])
				best = cs.live[s], slot = s;
		if (best > SEG - SEG / 16)
			return ENOSPC;
		if (spill_io(0, cs.gcbuf, (cs.used[slot] + BLK - 1) & ~(size_t)(BLK - 1),
			     (uint64_t)(slot - cs.nseg) * SEG))
			return EIO;
		out = survivors(cs.gcbuf, slot, cs.evbuf, 0, SEG, NULL);
		if (cs.state[slot] == SEG_BAD)
			return EIO; /* not all of it could be found, so it must not be written over */
	} else { /* a free one that nobody is still reading: bring it to the top of the stack */
		uint32_t k = cs.nsfree;
		while (k && cs.pins[cs.sfree[k - 1]])
			k--;
		if (!k)
			return ENOSPC;
		slot = cs.sfree[k - 1];
		cs.sfree[k - 1] = cs.sfree[cs.nsfree - 1];
		cs.sfree[cs.nsfree - 1] = slot;
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
	if (reuse) {
		/* its records are being read from the file as we speak: all of this under the lock */
		err = spill_io(1, cs.evbuf, (out + BLK - 1) & ~(size_t)(BLK - 1), (uint64_t)(slot - cs.nseg) * SEG);
	} else {
		/*
		 * A free segment holds nothing anyone can ask for, so the write to
		 * disk, the slow part, is done with the lock dropped. Meanwhile the
		 * records are still on the card and may be read, discarded or
		 * shared; whichever have gone by the time the lock is back are
		 * simply not moved.
		 */
		cs.nsfree--;
		cs.state[slot] = SEG_CLEANING;
		pthread_mutex_unlock(&cs.lock);
		err = spill_io(1, cs.evbuf, (out + BLK - 1) & ~(size_t)(BLK - 1), (uint64_t)(slot - cs.nseg) * SEG);
		pthread_mutex_lock(&cs.lock);
	}
	if (err) {
		if (!reuse)
			seg_free(slot);
		return EIO;
	}
	cs.state[slot] = SEG_CLEANING; /* its own records leaving must not free it */
	size_t k = 0, went = 0;
	for (size_t pos = 0; pos < out; pos += REC_HDR + r.len, k++) {
		memcpy(&r, cs.evbuf + pos, sizeof r);
		/*
		 * Gone, or gone elsewhere, while we were writing? The nonce is
		 * compared too: the id and the place may both have been given
		 * to a new record in the meantime, and what we wrote is the old.
		 */
		if (!cs.rloc[r.id] || E_OFF(cs.rloc[r.id]) != moved_from[k] || cs.rnon[r.id] != (uint32_t)r.nonce)
			continue;
		rec_place(r.id, E_MAKE((uint64_t)slot * SEG + pos, r.len));
		went += REC_HDR + r.len;
	}
	cs.used[slot] = out;
	cs.hits[slot] = 0;
	held_drop(slot);
	cs.seq[slot] = ++cs.clock;
	cs.state[slot] = SEG_CLOSED;
	if (!cs.live[slot])
		seg_free(slot);
	cs.evicted += went;
	return 0;
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
	int whole = cs.state[v] != SEG_BAD;
	if (whole)
		cs.state[v] = SEG_CLEANING; /* keep it out of the free list until its records have moved */
	int err = log_append(cs.gcbuf, out);
	if (err) {
		if (whole)
			cs.state[v] = SEG_CLOSED;
		return err;
	}
	if (!whole)
		return 0; /* what could be found has moved; the rest stays, and so does the segment */
	seg_free(v);
	cs.cleaned++;
	return 0;
}

/* Bring the most recently spilled segment back onto the card. */
static void rehydrate(void)
{
	uint32_t v = 0;
	uint64_t newest = 0;

	for (uint32_t s = cs.nseg; s < cs.nseg + cs.nspill; s++)
		if (cs.state[s] == SEG_CLOSED && cs.seq[s] > newest)
			newest = cs.seq[s], v = s;
	if (!newest)
		return;
	size_t span = (cs.used[v] + BLK - 1) & ~(size_t)(BLK - 1);
	cs.pins[v]++; /* nothing may rewrite it while we read it with the lock dropped */
	pthread_mutex_unlock(&cs.lock);
	int ok = !spill_io(0, cs.rebuf, span, (uint64_t)(v - cs.nseg) * SEG);
	pthread_mutex_lock(&cs.lock);
	cs.pins[v]--;
	if (ok && cs.state[v] == SEG_CLOSED && cs.seq[v] == newest && cs.nfree >= 4) {
		size_t out = survivors(cs.rebuf, v, cs.rebuf, 0, SEG, NULL);
		if (!log_append(cs.rebuf, out))
			cs.returned += out;
	}
}

/*
 * The keeper: the one thread that does the housekeeping, so that a write
 * never has to. A write goes to the card and nowhere else; the keeper's job
 * is to see that there is always room there for it. It keeps a few free
 * segments in hand by cleaning out those with dead records in them; with a
 * spill file it also holds the card below its high-water mark by moving the
 * oldest records out, and below the low-water mark brings them back. A write
 * waits for it only when the free segments have all gone, which means writes
 * are arriving faster than the disk can take what they displace.
 */
static void *keeper_thread(void *arg)
{
	(void)arg;
	cu.CtxSetCurrent(ctx);
	prctl(PR_SET_IO_FLUSHER, 1, 0, 0, 0);
	pthread_mutex_lock(&cs.lock);
	for (;;) {
		uint64_t on_card = cs.bytes - cs.sbytes;
		int spill = spill_fd >= 0, err = EAGAIN;

		if (spill && on_card * 100 > phys_size * high_pct) {
			if ((err = evict()) && cs.nfree < cs.ahead)
				err = clean_one(); /* the spill is full, or failing: at least tidy up */
		} else if (cs.nfree < cs.ahead) {
			if ((err = clean_one()))
				err = evict();
		} else if (spill && cs.sbytes && on_card * 100 < phys_size * low_pct && cs.nfree >= 4) {
			rehydrate();
			err = 0;
		}
		/*
		 * Nothing could be freed, whether for want of room or because the
		 * card or the disk gave an error, and writes are waiting or soon
		 * will be: they are refused. Waiting for ever would hang the
		 * machine; a refused write leaves the page in memory.
		 */
		if (err && err != EAGAIN && (cs.nfree < 3 || cs.waiting))
			cs.stuck = 1;
		pthread_cond_broadcast(&cs.room);
		if (!err) { /* let those who want the lock have it between one step and the next */
			pthread_mutex_unlock(&cs.lock);
			sched_yield();
			pthread_mutex_lock(&cs.lock);
		}
		if (err) { /* nothing to do, or nothing that can be done: wait to be asked */
			struct timespec until;
			clock_gettime(CLOCK_REALTIME, &until);
			until.tv_nsec += 50000000;
			until.tv_sec += until.tv_nsec / 1000000000, until.tv_nsec %= 1000000000;
			pthread_cond_timedwait(&cs.wake, &cs.lock, &until);
		}
	}
	return NULL;
}

/*
 * See that there is room on the card for `need` bytes of records, leaving two
 * segments for the keeper to work in. There nearly always is; if not, wait
 * for the keeper, and give up only when it says nothing more can be freed.
 */
static int make_room(size_t need)
{
	int waited = 0;

	for (;;) {
		uint64_t avail = (uint64_t)cs.nfree * SEG + SEG - cs.used[cs.open];
		if (cs.nfree < cs.ahead ||
		    (spill_fd >= 0 && (cs.bytes - cs.sbytes) * 100 > phys_size * high_pct))
			pthread_cond_signal(&cs.wake);
		if (avail >= need + 2 * SEG)
			return 0;
		if (cs.stuck) {
			cs.refused++;
			return ENOSPC;
		}
		cs.waits += !waited++;
		cs.waiting++;
		pthread_cond_signal(&cs.wake);
		pthread_cond_wait(&cs.room, &cs.lock);
		cs.waiting--;
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
	enum { ZERO, SHARED, NEW };
	uint64_t lpn = off / BLK;
	struct fp print[MAX_PAGES];
	uint32_t id[MAX_PAGES];
	uint8_t kind[MAX_PAGES];
	size_t n = len / BLK, out = 0, fresh = 0;
	int err = 0;

	if (off % BLK || len % BLK || n > MAX_PAGES)
		return EINVAL;
	ct_init();
	for (size_t i = 0; i < n; i++) {
		kind[i] = page_is_zero(buf + i * BLK) ? ZERO : NEW;
		if (kind[i] == NEW) {
			fresh++;
			if (page_print(buf + i * BLK, &print[i]))
				return EIO;
		}
	}
	/* A page we already hold a copy of just points at that copy. */
	if (fresh) {
		pthread_mutex_lock(&cs.lock);
		for (size_t i = 0; i < n; i++) {
			uint32_t r;
			if (kind[i] != NEW)
				continue;
			r = idx_find(&print[i]);
			for (size_t j = 0; j < i && !r; j++) /* or one arriving in this same request */
				if (kind[j] == NEW && print[j].a == print[i].a && print[j].b == print[i].b)
					r = id[j];
			if (r && cs.rref[r] != UINT32_MAX) {
				cs.rref[r]++;
				id[i] = r;
				kind[i] = SHARED;
				fresh--;
			} else if (!(id[i] = rec_alloc(&print[i]))) { /* the rest get a record of their own */
				err = ENOSPC;
			}
		}
		pthread_mutex_unlock(&cs.lock);
	}
	for (size_t i = 0; i < n && fresh && !err; i++) {
		const char *p = buf + i * BLK;
		if (kind[i] != NEW)
			continue;
		size_t c = z.Compress(ct.cc, ct.stage + out + REC_HDR, RAW_ABOVE, p, BLK, clevel);
		if (z.IsError(c) || !c || c > RAW_ABOVE) {
			memcpy(ct.stage + out + REC_HDR, p, BLK);
			c = BLK;
		}
		struct rec r = { .id = id[i], .len = c };
		if (rec_crypt(1, &r, ct.stage + out + REC_HDR))
			err = EIO;
		memcpy(ct.stage + out, &r, sizeof r);
		out += REC_HDR + c;
	}
	pthread_mutex_lock(&cs.lock);
	if (out && !err && !(err = make_room(out))) {
		idx_reserve(fresh);
		err = log_append(ct.stage, out);
	}
	for (size_t i = 0; i < n; i++) {
		if (kind[i] == ZERO) {
			if (!err)
				map_drop(lpn + i);
		} else if (err) {
			if (id[i])
				rec_unref(id[i]);
		} else {
			if (kind[i] == NEW)
				idx_insert(id[i]);
			map_set(lpn + i, id[i]);
		}
	}
	pthread_mutex_unlock(&cs.lock);
	return err;
}

/*
 * A record has failed its check. Neither the card's memory nor, usually, the
 * machine's corrects errors by itself, so before the page is given up for
 * lost: read the record again, which puts right a fault on the way here; and
 * if it is still wrong, try it with each of its bits turned over in turn,
 * which finds a single bit that has flipped where it lay. The authenticator
 * says when the right one has been found (a wrong one passing is a chance of
 * about 1 in 2^110), and the record is then put right where it is stored.
 * Some tens of milliseconds at worst, for something that should never happen.
 */
static int rescue(uint64_t lpn, char *p)
{
	struct rec r;
	uint64_t e, seq = 0, at = 0;
	uint32_t id, non;
	size_t len, total, skew = 0;
	long bit = -1, nbits;
	int err, spilled, mended = 0;
	char *stored, *work = ct.work;

	pthread_mutex_lock(&cs.lock);
	id = cs.map[lpn], e = cs.rloc[id], non = cs.rnon[id];
	if (!e) { /* written again meanwhile, as zeros */
		pthread_mutex_unlock(&cs.lock);
		memset(p, 0, BLK);
		return 0;
	}
	len = E_LEN(e), total = REC_HDR + len, spilled = E_OFF(e) >= phys_size;
	if (!spilled)
		err = vram_rw(0, E_OFF(e), ct.fix, total);
	else {
		at = (E_OFF(e) - phys_size) & ~(uint64_t)(BLK - 1), skew = E_OFF(e) - phys_size - at;
		seq = cs.seq[E_OFF(e) / SEG];
		err = spill_io(0, ct.fix, (skew + total + BLK - 1) & ~(size_t)(BLK - 1), at);
	}
	pthread_mutex_unlock(&cs.lock);
	stored = ct.fix + skew, nbits = (long)total * 8;
	for (; !err; bit++) {
		if (bit == nbits) {
			err = EIO;
			break;
		}
		memcpy(work, stored, total);
		if (bit >= 0)
			work[bit >> 3] ^= 1 << (bit & 7);
		memcpy(&r, work, sizeof r);
		if (r.id == id && r.len == len && (uint32_t)r.nonce == non && !rec_crypt(0, &r, work + REC_HDR))
			break;
	}
	if (!err && len < BLK && z.Decompress(ct.dc, p, BLK, work + REC_HDR, len) != BLK)
		err = EIO;
	else if (!err && len == BLK)
		memcpy(p, work + REC_HDR, BLK);

	pthread_mutex_lock(&cs.lock);
	if (err)
		cs.lost++;
	else if (bit < 0)
		cs.reread++;
	else {
		cs.mended++;
		stored[bit >> 3] ^= 1 << (bit & 7);
		/* put it right where it lies, unless it has moved or gone since */
		if (cs.rloc[id] == e && cs.rnon[id] == non && !spilled)
			mended = !vram_rw(1, E_OFF(e) + (bit >> 3), stored + (bit >> 3), 1);
		else if (cs.rloc[id] == e && cs.rnon[id] == non && cs.seq[E_OFF(e) / SEG] == seq) {
			size_t blk = (skew + (bit >> 3)) & ~(size_t)(BLK - 1);
			mended = !spill_io(1, ct.fix + blk, BLK, at + blk);
			for (int k = 0; k < 16; k++)
				if (cs.held[k].valid && cs.held[k].seg == E_OFF(e) / SEG)
					cs.held[k].valid = 0;
		}
	}
	pthread_mutex_unlock(&cs.lock);
	if (say()) {
		const char *where = spilled ? "on the spill disk" : "in VRAM";
		if (err)
			fprintf(stderr, "vroom: stored page %llu (%s) has been altered or damaged beyond repair\n",
				(unsigned long long)lpn, where);
		else if (bit < 0)
			fprintf(stderr, "vroom: stored page %llu (%s) read wrong once and right the second time\n",
				(unsigned long long)lpn, where);
		else
			fprintf(stderr, "vroom: stored page %llu (%s) had one bit flipped: found%s\n",
				(unsigned long long)lpn, where, mended ? " and put right" : "");
	}
	return err ? EIO : 0;
}

static int cs_read(uint64_t off, char *buf, size_t len)
{
	uint64_t lpn = off / BLK;
	size_t n = len / BLK, in = 0, nsp = 0;
	uint32_t pos[MAX_PAGES], clen[MAX_PAGES], non[MAX_PAGES];
	struct {
		uint64_t at, start, end;
		size_t to, span;
	} sp[MAX_PAGES]; /* reads of the spill file, done once the lock is dropped */
	struct held *load = NULL; /* a whole segment this request reads to keep */
	uint64_t load_from = 0;
	size_t load_to = 0, load_len = 0, load_span = 0;
	int bad = 0;

	if (off % BLK || len % BLK || n > MAX_PAGES)
		return EINVAL;
	ct_init();
	pthread_mutex_lock(&cs.lock);
	for (size_t i = 0; i < n && !bad;) {
		uint64_t e = cs.rloc[cs.map[lpn + i]], start, end;
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
			non[i] = cs.rnon[cs.map[lpn + i]];
			end += REC_HDR + E_LEN(e);
		} while (++i < n && (e = cs.rloc[cs.map[lpn + i]]) && E_OFF(e) == end && end != phys_size);
		if (start < phys_size) {
			bad = vram_rw(0, start, ct.stage + in, end - start);
			in += end - start;
			continue;
		}
		uint32_t g = start / SEG;
		uint64_t in_seg = start - (uint64_t)g * SEG;
		if ((end - 1) / SEG == g) {
			struct held *h = NULL, *spare = NULL;
			for (int k = 0; k < 16; k++) {
				struct held *c = &cs.held[k];
				if (c->valid && c->seg == g && c->seq == cs.seq[g])
					h = c;
				if (!c->loading && (!spare || c->used < spare->used))
					spare = c;
			}
			if (h) { /* already in memory */
				memcpy(ct.stage + in, h->buf + in_seg, end - start);
				h->used = ++cs.held_clock;
				cs.held_hits++;
				in += end - start;
				continue;
			}
			/* Three reads in a row each a little further on: someone is reading through it. */
			if (in_seg >= cs.lastoff[g] && in_seg - cs.lastoff[g] <= 131072 && cs.hits[g] < 255)
				cs.hits[g]++;
			else
				cs.hits[g] = 1;
			cs.lastoff[g] = end - (uint64_t)g * SEG;
			/* a read that already takes in a good stretch gains nothing from more */
			if (cs.hits[g] >= 3 && !load && spare && end - start <= 65536) {
				load = spare;
				*load = (struct held){ .seg = g, .seq = cs.seq[g], .loading = 1, .buf = spare->buf };
				load_from = in_seg, load_to = in, load_len = end - start;
				load_span = (cs.used[g] + BLK - 1) & ~(size_t)(BLK - 1);
				cs.pins[g]++;
				in += end - start;
				continue;
			}
		}
		/*
		 * In the spill file: whole blocks, into a block-aligned place, and
		 * not while holding the lock, which everything else needs. What is
		 * read cannot change meanwhile: records are never altered where
		 * they lie, and a segment with a reader is not rewritten.
		 */
		uint64_t fo = start - phys_size, a = fo & ~(uint64_t)(BLK - 1);
		size_t al = (in + BLK - 1) & ~(size_t)(BLK - 1), shift = al + (fo - a) - in;
		for (size_t k = first; k < i; k++)
			pos[k] += shift;
		sp[nsp].at = a, sp[nsp].to = al, sp[nsp].start = start, sp[nsp].end = end;
		sp[nsp].span = ((fo + (end - start) + BLK - 1) & ~(uint64_t)(BLK - 1)) - a;
		for (uint64_t s2 = start / SEG; s2 <= (end - 1) / SEG; s2++)
			cs.pins[s2]++;
		in = al + sp[nsp++].span;
	}
	pthread_mutex_unlock(&cs.lock);
	for (size_t k = 0; k < nsp; k++)
		if (!bad)
			bad = spill_io(0, ct.stage + sp[k].to, sp[k].span, sp[k].at);
	if (load) {
		int failed = spill_io(0, load->buf, load_span, (uint64_t)(load->seg - cs.nseg) * SEG);
		if (!failed)
			memcpy(ct.stage + load_to, load->buf + load_from, load_len);
		bad |= failed;
		pthread_mutex_lock(&cs.lock);
		cs.pins[load->seg]--;
		load->loading = 0;
		load->valid = !failed && load->seq == cs.seq[load->seg] && cs.state[load->seg] == SEG_CLOSED;
		load->used = ++cs.held_clock;
		pthread_mutex_unlock(&cs.lock);
	}
	if (nsp) {
		pthread_mutex_lock(&cs.lock);
		for (size_t k = 0; k < nsp; k++)
			for (uint64_t g = sp[k].start / SEG; g <= (sp[k].end - 1) / SEG; g++)
				cs.pins[g]--;
		pthread_mutex_unlock(&cs.lock);
	}
	if (bad)
		return EIO;
	for (size_t i = 0; i < n; i++) {
		char *p = buf + i * BLK, *data = ct.stage + pos[i];
		struct rec r;

		if (!clen[i]) {
			memset(p, 0, BLK);
			continue;
		}
		memcpy(&r, data - REC_HDR, sizeof r);
		/*
		 * The nonce must be the one this page was last sealed with: a
		 * record can be genuine and still be the wrong one, an older
		 * writing of the same page put back by a failing disk or by
		 * someone who can write to it.
		 */
		if (r.len != clen[i] || (uint32_t)r.nonce != non[i] || rec_crypt(0, &r, data)) {
			if (rescue(lpn + i, p))
				return EIO;
			continue;
		}
		if (r.len == BLK)
			memcpy(p, data, BLK);
		else if (z.Decompress(ct.dc, p, BLK, data, r.len) != BLK) {
			if (say())
				fprintf(stderr, "vroom: stored page %llu does not decompress\n",
					(unsigned long long)(lpn + i));
			return EIO;
		}
	}
	return 0;
}

/* Would reading this have to wait for the spill file? */
static int cs_reads_spill(uint64_t off, size_t len)
{
	int yes = 0;
	if (spill_fd < 0 || off % BLK)
		return 0;
	pthread_mutex_lock(&cs.lock);
	for (uint64_t p = off / BLK; p < (off + len) / BLK && !yes; p++)
		yes = cs.map[p] && E_OFF(cs.rloc[cs.map[p]]) >= phys_size;
	pthread_mutex_unlock(&cs.lock);
	return yes;
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
	uint64_t t0 = perf_begin(wr ? P_REQ_WRITE : P_REQ_READ, len);
	int err;

	if (counting && !(off % BLK))
		count_io(wr, off, buf, len);
	if (!cratio)
		err = vram_rw(wr, off, buf, len) ? EIO : 0;
	else
		err = wr ? cs_write(off, buf, len) : cs_read(off, buf, len);
	perf_end(wr ? P_REQ_WRITE : P_REQ_READ, len, t0);
	return err;
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
/* The time below which the fraction q of the timed operations between two readings finished. */
static double hist_quantile(const struct perf *now, const struct perf *then, double q)
{
	uint64_t n = now->t_ops - then->t_ops, seen = 0;
	if (!n)
		return 0;
	for (unsigned i = 0; i < HIST; i++) {
		seen += now->hist[i] - then->hist[i];
		if (seen >= q * n)
			return hist_ns(i);
	}
	return hist_ns(HIST - 1);
}

/*
 * While serving: every two seconds write /run/vroom/DEVICE.stat for vroomctl,
 * and warn when the store is nearly full. Readings are kept for five minutes
 * so that rates and latencies can be given for the last minute and the last
 * five; the peak is the busiest two seconds since the start.
 */
#define TICK 2
#define KEPT (300 / TICK + 1)
static void *stat_thread(void *arg)
{
	static const char *kind[P_KINDS] = { "req_read", "req_write", "vram_read", "vram_write",
					     "disk_read", "disk_write" };
	static const int window[2] = { 60, 300 };
	static struct perf ring[KEPT][P_KINDS];
	char path[128], tmp[136];
	const char *base = strrchr(dev_path, '/');
	double peak[P_KINDS] = { 0 };
	uint64_t started = now_ns();
	int warned = 0;

	(void)arg;
	snprintf(path, sizeof path, "/run/vroom/%s.stat", base ? base + 1 : dev_path);
	snprintf(tmp, sizeof tmp, "%s.tmp", path);
	for (unsigned tick = 0;; tick++, sleep(TICK)) {
		pthread_mutex_lock(&cs.lock);
		uint64_t pages = cs.pages, bytes = cs.bytes - cs.sbytes, refused = cs.refused, sbytes = cs.sbytes;
		uint64_t recs = cs.recs, waits = cs.waits, reread = cs.reread, mended = cs.mended, lost = cs.lost;
		uint64_t held = (uint64_t)(cs.nseg - cs.nfree) * SEG;
		pthread_mutex_unlock(&cs.lock);
		int pct = bytes * 100 / phys_size;

		if (spill_fd >= 0)
			pct = spill_size ? sbytes * 100 / spill_size : 0;
		if (pct >= 90 && !warned)
			fprintf(stderr, "vroom: WARNING: the %s is %d%% full of compressed data; "
					"writes will fail when it is full\n",
				spill_fd >= 0 ? "spill" : "VRAM", pct);
		warned = pct >= 90 ? 1 : pct < 80 ? 0 : warned;

		struct perf *cur = ring[tick % KEPT];
		memcpy(cur, perf, sizeof perf);
		if (tick)
			for (int k = 0; k < P_KINDS; k++) {
				double r = (cur[k].bytes - ring[(tick - 1) % KEPT][k].bytes) / (double)TICK;
				if (r > peak[k])
					peak[k] = r;
			}
		FILE *f = fopen(tmp, "w");
		if (!f)
			continue;
		fchmod(fileno(f), 0600); /* how many pages share a copy says something about what they hold */
		if (cratio)
			fprintf(f, "COMPRESSED=1\nPAGES=%llu\nSTORED_BYTES=%llu\nSEGMENT_BYTES=%llu\nREFUSED=%llu\n"
				   "SPILLED_BYTES=%llu\nSPILL_BYTES=%llu\nRECORDS=%llu\nWAITS=%llu\n"
				   "REREAD=%llu\nMENDED=%llu\nUNREADABLE=%llu\n",
				(unsigned long long)pages, (unsigned long long)bytes, (unsigned long long)held,
				(unsigned long long)refused, (unsigned long long)sbytes,
				(unsigned long long)spill_size, (unsigned long long)recs,
				(unsigned long long)waits, (unsigned long long)reread,
				(unsigned long long)mended, (unsigned long long)lost);
		fprintf(f, "VRAM_BYTES=%llu\nCIPHER=%s\nUPTIME=%llu\nSAMPLE=%d\n", (unsigned long long)phys_size,
			cratio ? cipher_name : "none", (unsigned long long)((now_ns() - started) / 1000000000),
			SAMPLE);
		/*
		 * One line for each kind of operation:
		 * #perf KIND total-ops total-bytes peak-bytes/s, then for each window
		 * its length in seconds, ops/s, bytes/s and the p50, p95 and p99 time in ns;
		 * last, the bytes and the nanoseconds of the operations that were timed.
		 */
		for (int k = 0; k < P_KINDS; k++) {
			fprintf(f, "#perf %s %llu %llu %.0f", kind[k], (unsigned long long)cur[k].ops,
				(unsigned long long)cur[k].bytes, peak[k]);
			for (int w = 0; w < 2; w++) {
				unsigned back = (unsigned)window[w] / TICK < tick ? (unsigned)window[w] / TICK : tick;
				struct perf *old = &ring[(tick - back) % KEPT][k];
				double secs = back * TICK;
				fprintf(f, " %.0f %.1f %.0f %.0f %.0f %.0f", secs,
					secs ? (cur[k].ops - old->ops) / secs : 0,
					secs ? (cur[k].bytes - old->bytes) / secs : 0,
					hist_quantile(&cur[k], old, 0.50), hist_quantile(&cur[k], old, 0.95),
					hist_quantile(&cur[k], old, 0.99));
			}
			fprintf(f, " %llu %llu\n", (unsigned long long)cur[k].t_bytes, (unsigned long long)cur[k].t_ns);
		}
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
	thread_ready();

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
		int err = off > dev_size || len > dev_size - off ? EINVAL : 0;
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

/*
 * Is something using the device that the two lists do not show: a partition
 * on it, LVM or dm-crypt on top of it? The kernel refuses an exclusive open
 * of a device that anything has claimed.
 */
static int held_open(void)
{
	int fd = open(dev_path, O_RDONLY | O_EXCL | O_CLOEXEC);
	if (fd >= 0)
		close(fd);
	return fd < 0 && errno == EBUSY;
}

/* Returns once a stop signal arrives while the device holds nothing live. */
static void *sig_thread(void *arg)
{
	sigset_t *set = arg;
	int sig;
	for (;;) {
		sigwait(set, &sig);
		int sw = listed_in("/proc/swaps"), mnt = !sw && listed_in("/proc/mounts");
		if (sw || mnt || held_open()) {
			fprintf(stderr, "vroom: ignoring signal %d: %s is %s. %s%s%s first.\n", sig, dev_path,
				sw ? "active swap" : mnt ? "mounted" : "in use",
				sw ? "Run 'swapoff " : mnt ? "Run 'umount " : "Release it",
				sw || mnt ? dev_path : "", sw || mnt ? "'" : "");
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
	/* Every signal that would otherwise end us on the spot, not only the usual two. */
	sigemptyset(&stop_set);
	sigaddset(&stop_set, SIGINT);
	sigaddset(&stop_set, SIGTERM);
	sigaddset(&stop_set, SIGHUP);
	sigaddset(&stop_set, SIGQUIT);
	sigaddset(&stop_set, SIGUSR1);
	sigaddset(&stop_set, SIGUSR2);
	pthread_sigmask(SIG_BLOCK, &stop_set, NULL);
	signal(SIGPIPE, SIG_IGN);
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
	/* A log that is not being read must lose lines, not stop us (when it is the journal's socket). */
	struct stat st;
	if (!fstat(2, &st) && S_ISSOCK(st.st_mode))
		fcntl(2, F_SETFL, fcntl(2, F_GETFL) | O_NONBLOCK);
	/* Tell vroomctl the device is this process's, and may be put to use. */
	char path[128];
	const char *base = strrchr(dev_path, '/');
	snprintf(path, sizeof path, "/run/vroom/%s.pid", base ? base + 1 : dev_path);
	if ((f = fopen(path, "w"))) {
		fprintf(f, "%d\n", getpid());
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
		spawn(worker, (void *)(intptr_t)sp[1], &t);
		close(sp[0]); /* the kernel holds its own reference */
	}
	await_threads(threads);
	spawn(sig_thread, &stop_set, &t);
	spawn(stat_thread, NULL, &t);
	if (counting)
		spawn(count_thread, NULL, &t);
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

/* Queue a multishot poll: a completion each time fd becomes readable. */
static void ring_poll(struct ring *r, int fd, uint64_t ud)
{
	unsigned idx = r->tail & *r->sq_mask;
	struct io_uring_sqe *s = (struct io_uring_sqe *)(r->sqes + ((size_t)idx << r->shift));
	memset(s, 0, (size_t)1 << r->shift);
	s->opcode = IORING_OP_POLL_ADD;
	s->fd = fd;
	s->poll32_events = POLLIN;
	s->len = IORING_POLL_ADD_MULTI;
	s->user_data = ud;
	r->sq_array[idx] = idx;
	r->tail++;
	r->pending++;
}

#define UBLK_DEPTH 32
#define UBLK_IO_BYTES (256 * 1024)
#define POLL_UD UINT64_MAX
#define HELPERS 6

/*
 * A queue is one thread, and what it does itself holds up everything behind
 * it. So it keeps only what is quick: reads of what is on the card, which is
 * what a stopped process is usually waiting for. A read that has to go to the
 * spill file takes the disk's time, and a write to a compressed device takes
 * a core's time to compress and encrypt; both are handed to helper threads,
 * reads ahead of writes. A helper says it has finished through the queue's
 * eventfd, and the queue completes the request.
 */
struct slow {
	int q, wr;
	unsigned tag;
	uint64_t off;
	uint32_t len;
	int res;
	char *buf;
	struct slow *next;
};
static struct {
	pthread_mutex_t lock;
	pthread_cond_t more;
	struct slow *head[2], *tail[2]; /* reads, writes */
} slowq = { .lock = PTHREAD_MUTEX_INITIALIZER, .more = PTHREAD_COND_INITIALIZER };
static struct {
	pthread_mutex_t lock;
	struct slow *done;
	int efd;
} qdone[MAX_THREADS];

static void *slow_reader(void *arg)
{
	(void)arg;
	cu.CtxSetCurrent(ctx);
	prctl(PR_SET_IO_FLUSHER, 1, 0, 0, 0);
	thread_ready();
	for (;;) {
		pthread_mutex_lock(&slowq.lock);
		while (!slowq.head[0] && !slowq.head[1])
			pthread_cond_wait(&slowq.more, &slowq.lock);
		int w = !slowq.head[0]; /* a read, if one is waiting */
		struct slow *j = slowq.head[w];
		if (!(slowq.head[w] = j->next))
			slowq.tail[w] = NULL;
		pthread_mutex_unlock(&slowq.lock);

		int err = dev_rw(j->wr, j->off, j->buf, j->len);
		j->res = err ? -err : (int)j->len;
		pthread_mutex_lock(&qdone[j->q].lock);
		j->next = qdone[j->q].done;
		qdone[j->q].done = j;
		pthread_mutex_unlock(&qdone[j->q].lock);
		uint64_t one = 1;
		if (write(qdone[j->q].efd, &one, sizeof one) < 0 && say())
			fprintf(stderr, "vroom: waking queue %d: %s\n", j->q, strerror(errno));
	}
	return NULL;
}
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
	struct slow *slow = calloc(UBLK_DEPTH, sizeof *slow);
	int handoff = cratio != 0;

	cu.CtxSetCurrent(ctx);
	prctl(PR_SET_IO_FLUSHER, 1, 0, 0, 0);
	if (cratio)
		ct_init();
	ring_init(&ring, 2 * UBLK_DEPTH,
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
	if (handoff) {
		pthread_mutex_init(&qdone[q].lock, NULL);
		qdone[q].efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
		if (qdone[q].efd < 0 || !slow)
			die("ublk: eventfd: %s", strerror(errno));
		ring_poll(&ring, qdone[q].efd, POLL_UD);
	}

	for (unsigned live = UBLK_DEPTH; live;) {
		if ((r = ring_enter(&ring, 1)))
			die("ublk: io_uring_enter: %s", strerror(-r));
		unsigned first = *ring.cq_head, tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE), head;
		/* Everything that has arrived, twice over: reads and the rest first, then the writes. */
		for (int writes = 0; writes < 2; writes++)
		for (head = first; head != tail; head++) {
			struct io_uring_cqe *e = &ring.cqes[head & *ring.cq_mask];
			unsigned tag = e->user_data;
			int is_write = e->user_data != POLL_UD && e->res == UBLK_IO_RES_OK &&
				       (iod[tag].op_flags & 0xff) == UBLK_IO_OP_WRITE;
			if (is_write != writes)
				continue;
			if (e->user_data == POLL_UD) { /* helpers have finished some requests */
				uint64_t v;
				if (read(qdone[q].efd, &v, sizeof v) < 0 && errno != EAGAIN)
					die("ublk: eventfd: %s", strerror(errno));
				pthread_mutex_lock(&qdone[q].lock);
				struct slow *j = qdone[q].done, *nx;
				qdone[q].done = NULL;
				pthread_mutex_unlock(&qdone[q].lock);
				for (; j; j = nx) {
					nx = j->next;
					struct ublksrv_io_cmd c = { .q_id = q, .tag = j->tag, .result = j->res,
								    .addr = (uintptr_t)j->buf };
					ring_cmd(&ring, cdev_fd, UBLK_U_IO_COMMIT_AND_FETCH_REQ, &c, sizeof c, j->tag);
				}
				if (!(e->flags & IORING_CQE_F_MORE))
					ring_poll(&ring, qdone[q].efd, POLL_UD);
				continue;
			}
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
			if (handoff && len <= ublk_io_bytes && off + len <= dev_size &&
			    (is_write || ((d->op_flags & 0xff) == UBLK_IO_OP_READ && cs_reads_spill(off, len)))) {
				struct slow *j = &slow[tag];
				*j = (struct slow){ .q = q, .wr = is_write, .tag = tag, .off = off, .len = len,
						    .buf = buf[tag] };
				pthread_mutex_lock(&slowq.lock);
				if (slowq.tail[is_write])
					slowq.tail[is_write]->next = j;
				else
					slowq.head[is_write] = j;
				slowq.tail[is_write] = j;
				pthread_cond_signal(&slowq.more);
				pthread_mutex_unlock(&slowq.lock);
				continue; /* completed when a helper reports back */
			}
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

	for (int i = 0; cratio && i < HELPERS; i++)
		spawn(slow_reader, NULL, NULL);
	if (cratio)
		await_threads(HELPERS);
	for (int i = 0; i < threads; i++)
		spawn(ublk_queue, (void *)(intptr_t)i, &tq[i]);
	/* Returns once every queue has handed all its tags to the kernel. */
	if ((r = ublk_ctrl(UBLK_U_CMD_START_DEV, NULL, 0, getpid())))
		die("ublk: start: %s", strerror(-r));
	harden();
	fprintf(stderr, "vroom: serving %llu MiB on %s (ublk, %d queues)\n",
		(unsigned long long)(dev_size / MiB), dev_path, threads);

	spawn(sig_thread, &stop_set, &ts);
	spawn(stat_thread, NULL, NULL);
	if (counting)
		spawn(count_thread, NULL, NULL);
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

	if (kind == 2) /* one of 500 pages that turn up all over the device */
		blk = (x >> 20) % 500, x = blk * 0x9E3779B97F4A7C15ULL | 1, q = 1024;

	if (!kind || ver == 254) { /* 254: discarded */
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

/*
 * The store's tables must agree with each other. Checked between the stages
 * of the self-test, with the lock held so that the keeper is between steps.
 */
static void cs_check(const char *when)
{
	uint64_t npages = dev_size / BLK, live = 0, slive = 0, placed = 0, indexed = 0, pages = 0, unused = 0;
	uint32_t *refs = calloc(cs.rmax + 1, sizeof *refs), *sl = calloc(cs.nseg + cs.nspill, sizeof *sl);
	uint8_t *seen = calloc(cs.rmax + 1, 1);
	const char *bad = NULL;

	pthread_mutex_lock(&cs.lock);
	for (uint64_t p = 0; p < npages; p++)
		if (cs.map[p])
			refs[cs.map[p]]++, pages++;
	for (uint64_t i = 0; i <= cs.idx_mask; i++) {
		uint32_t id = cs.idx[i];
		if (!id || id == IDX_GONE)
			continue;
		if (!cs.rloc[id])
			bad = "the index names a record that holds nothing";
		else if (seen[id]++)
			bad = "a record is in the index twice";
		indexed++;
	}
	for (uint32_t id = 1; id <= cs.rmax; id++) {
		uint64_t e = cs.rloc[id];
		if (!e) {
			if (refs[id])
				bad = "a page points at a record that holds nothing";
			continue;
		}
		placed++;
		if (refs[id] != cs.rref[id])
			bad = "a record's count of pages is wrong";
		if (!seen[id])
			bad = "a record is missing from the index";
		sl[E_OFF(e) / SEG] += REC_HDR + E_LEN(e);
	}
	for (uint32_t s = 0; s < cs.nseg + cs.nspill; s++) {
		if (sl[s] != cs.live[s])
			bad = "a segment's count of live bytes is wrong";
		if (cs.live[s] && cs.state[s] == SEG_FREE)
			bad = "a free segment holds live records";
		live += sl[s];
		if (s >= cs.nseg)
			slive += sl[s];
	}
	for (uint32_t id = cs.rfree; id; id = cs.rref[id])
		if (++unused > cs.rmax)
			break;
	if (!bad && (live != cs.bytes || slive != cs.sbytes))
		bad = "the totals of bytes held are wrong";
	if (!bad && (placed != cs.recs || indexed != placed || pages != cs.pages))
		bad = "the counts of pages and records are wrong";
	if (!bad && unused + cs.recs != cs.rmax)
		bad = "the list of unused records is wrong";
	pthread_mutex_unlock(&cs.lock);
	if (bad)
		die("selftest: %s, %s", bad, when);
	free(refs), free(sl), free(seen);
}

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

/* Several writers at once, each in its own part of the device, each checking what it wrote. */
struct churn {
	int fd;
	uint64_t lo, n;
};

static void *churner(void *arg)
{
	struct churn *c = arg;
	char *a = malloc(64 * BLK), *b = malloc(64 * BLK);
	uint64_t x = 88172645463325252ULL + c->lo;

	for (int i = 0; i < 400; i++) {
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		unsigned cnt = 1 + (x >> 20) % 64;
		uint64_t blk = c->lo + x % (c->n - cnt);
		for (unsigned k = 0; k < cnt; k++) {
			tver[blk + k] = (tver[blk + k] + 1) % 200;
			pattern(a + k * BLK, blk + k, tver[blk + k]);
		}
		if (nbd_io(c->fd, NBD_CMD_WRITE, blk * BLK, a, cnt * BLK))
			die("selftest: write failed with several writers at once");
		if (i % 4 == 0 && nbd_io(c->fd, NBD_CMD_TRIM, (c->lo + x % (c->n - 8)) * BLK, NULL, 8 * BLK) == 0) {
			uint64_t d = c->lo + x % (c->n - 8);
			for (unsigned k = 0; k < 8; k++)
				tver[d + k] = 254; /* reads as zeros */
		}
		x ^= x << 13, x ^= x >> 7, x ^= x << 17;
		cnt = 1 + (x >> 20) % 64, blk = c->lo + x % (c->n - cnt);
		for (unsigned k = 0; k < cnt; k++)
			if (tver[blk + k] == 254)
				memset(a + k * BLK, 0, BLK);
			else
				pattern(a + k * BLK, blk + k, tver[blk + k]);
		if (nbd_io(c->fd, NBD_CMD_READ, blk * BLK, b, cnt * BLK) || memcmp(a, b, cnt * BLK))
			die("selftest: MISMATCH with several writers at once, near block %llu", (unsigned long long)blk);
	}
	free(a), free(b);
	return NULL;
}

static void check_all(int fd, const char *when)
{
	uint64_t npages = dev_size / BLK;
	for (uint64_t b = 0; b < npages; b += MAX_PAGES)
		check_pages(fd, b, npages - b < MAX_PAGES ? npages - b : MAX_PAGES, when);
	cs_check(when);
}

static void selftest_compress(int threads)
{
	int fds[MAX_THREADS];
	pthread_t tw[MAX_THREADS], tc[MAX_THREADS];
	uint64_t npages = dev_size / BLK, x = 2463534242ULL;
	double t;

	tver = calloc(npages, 1);
	void *al[2]; /* aligned: the test reads the spill file directly, which a disk under LVM insists on */
	if (posix_memalign(&al[0], BLK, BUFSZ) || posix_memalign(&al[1], BLK, BUFSZ))
		die("selftest: out of memory");
	ta = al[0], tb = al[1];
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
	printf("                  %7llu pages share a copy with another (%.1f%%)\n",
	       (unsigned long long)(cs.pages - cs.recs), (cs.pages - cs.recs) * 100.0 / npages);
	if (spill_fd >= 0)
		printf("                  %7llu MiB of that in VRAM, %llu MiB spilled to the file\n",
		       (unsigned long long)((cs.bytes - cs.sbytes) / MiB), (unsigned long long)(cs.sbytes / MiB));
	t = now();
	check_all(fd, "in sequential read");
	printf("sequential read   %7.0f MiB/s  (verified)\n", dev_size / MiB / (now() - t));

	/* What is on the card must not be the page, and altering it must be noticed, not returned. */
	for (uint64_t b = 0; b < npages; b++) {
		uint64_t e = cs.rloc[cs.map[b]];
		char was;
		if (!e || E_LEN(e) != BLK || E_OFF(e) >= phys_size)
			continue; /* looking for a page stored as it is, uncompressed */
		pattern(ta, b, tver[b]);
		if (vram_rw(0, E_OFF(e) + REC_HDR, tb, BLK) || !memcmp(ta, tb, BLK))
			die("selftest: page %llu is stored unencrypted", (unsigned long long)b);
		/* One bit flipped, in the data, the tag or the nonce: found, put right on the card, and counted. */
		static const unsigned spots[] = { REC_HDR + 100, 20, 15, REC_HDR + BLK - 1, 5 };
		double worst = 0;
		for (unsigned k = 0; k < sizeof spots / sizeof *spots; k++) {
			uint64_t at = E_OFF(e) + spots[k], before = cs.mended;
			char flipped;
			if (vram_rw(0, at, &was, 1))
				die("selftest: could not read the card");
			flipped = was ^ (1 << (k + 2));
			t = now();
			if (vram_rw(1, at, &flipped, 1) || nbd_io(fd, NBD_CMD_READ, b * BLK, tb, BLK) ||
			    memcmp(ta, tb, BLK))
				die("selftest: a flipped bit at byte %u of a record was not put right", spots[k]);
			if (now() - t > worst)
				worst = now() - t;
			if (vram_rw(0, at, &flipped, 1) || flipped != was || cs.mended != before + 1)
				die("selftest: a flipped bit was read past but not mended on the card");
		}
		/* Two bits is more than can be put right: that must be an error, never wrong data. */
		char two[2], twas[2];
		uint64_t lost = cs.lost;
		if (vram_rw(0, E_OFF(e) + REC_HDR + 100, twas, 2))
			die("selftest: could not read the card");
		two[0] = twas[0] ^ 1, two[1] = twas[1] ^ 0x80;
		if (vram_rw(1, E_OFF(e) + REC_HDR + 100, two, 2) ||
		    nbd_io(fd, NBD_CMD_READ, b * BLK, tb, BLK) != EIO || cs.lost != lost + 1)
			die("selftest: an altered page was not refused");
		if (vram_rw(1, E_OFF(e) + REC_HDR + 100, twas, 2))
			die("selftest: could not restore the page");
		check_pages(fd, b, 1, "after restoring an altered page");
		printf("encryption            ok (%s; an altered page is refused)\n", cipher_name);
		printf("repair                ok (one flipped bit is found and put right, in %.0f ms at worst)\n",
		       worst * 1000);
		break;
	}

	/*
	 * Rewrite everything a few times, many pages to a request, then free it
	 * all and write the same pages again: the index of fingerprints is
	 * rebuilt somewhere in the middle of a request, and what was freed must
	 * not be found by it afterwards.
	 */
	for (int round = 0; round < 4; round++)
		for (uint64_t b = 0; b + 64 <= npages; b += 64)
			if (put_pages(fd, b, 64, round < 3))
				die("selftest: rewrite failed at block %llu", (unsigned long long)b);
	check_all(fd, "after rewriting everything");
	for (uint64_t off = 0; off < dev_size; off += 1024 * MiB) {
		uint64_t n = dev_size - off < 1024 * MiB ? dev_size - off : 1024 * MiB;
		if (nbd_io(fd, NBD_CMD_TRIM, off, NULL, n))
			die("selftest: discard failed");
	}
	cs_check("after freeing everything");
	for (uint64_t b = 0; b < npages; b += MAX_PAGES)
		if (put_pages(fd, b, npages - b < MAX_PAGES ? npages - b : MAX_PAGES, 0))
			die("selftest: write failed at block %llu", (unsigned long long)b);
	check_all(fd, "after freeing everything and writing it again");
	printf("rewrite and reuse     ok\n");

	/* A length on the card that has rotted must cost that page and no other. */
	for (uint64_t b = 0; b < npages; b++) {
		uint64_t e = cs.rloc[cs.map[b]];
		uint32_t v = E_OFF(e) / SEG;
		uint16_t len, wrong;
		if (!e || E_OFF(e) >= phys_size || cs.state[v] != SEG_CLOSED || cs.rref[cs.map[b]] != 1)
			continue;
		len = E_LEN(e), wrong = len ^ 0x41; /* two bits: one would simply be put right */
		pthread_mutex_lock(&cs.lock);
		if (vram_rw(1, E_OFF(e) + 4, (char *)&wrong, 2) || vram_rw(0, (uint64_t)v * SEG, cs.gcbuf, cs.used[v]))
			die("selftest: could not damage a record");
		survivors(cs.gcbuf, v, cs.gcbuf, 0, SEG, NULL);
		int noticed = cs.state[v] == SEG_BAD;
		pthread_mutex_unlock(&cs.lock);
		if (!noticed || nbd_io(fd, NBD_CMD_READ, b * BLK, tb, BLK) != EIO)
			die("selftest: a damaged record was not noticed");
		pthread_mutex_lock(&cs.lock);
		if (vram_rw(1, E_OFF(e) + 4, (char *)&len, 2))
			die("selftest: could not restore a record");
		cs.state[v] = SEG_CLOSED;
		pthread_mutex_unlock(&cs.lock);
		check_all(fd, "after a record's header was damaged and restored");
		printf("damaged header        noticed; the segment is set aside, nothing else is lost\n");
		break;
	}

	struct churn ch[MAX_THREADS];
	for (int i = 0; i < threads; i++) {
		ch[i] = (struct churn){ fds[i], npages / threads * i, npages / threads };
		pthread_create(&tc[i], NULL, churner, &ch[i]);
	}
	for (int i = 0; i < threads; i++)
		pthread_join(tc[i], NULL);
	check_all(fd, "after several writers at once");
	printf("%d writers at once     ok\n", threads);

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
		for (uint64_t c = 0; c < npages; c++) { /* a page stored as it is, in the file, must not be the page */
			uint64_t e = cs.rloc[cs.map[c]], fo = E_OFF(e) - phys_size;
			if (!e || E_LEN(e) != BLK || E_OFF(e) < phys_size)
				continue;
			pattern(ta, c, tver[c]);
			if (pread(spill_fd, tb, 3 * BLK, spill_base + (fo & ~(uint64_t)(BLK - 1))) < BLK + REC_HDR ||
			    !memcmp(ta, tb + (fo & (BLK - 1)) + REC_HDR, BLK))
				die("selftest: page %llu is in the spill file unencrypted", (unsigned long long)c);
			printf("spill file            holds no plain page\n");
			{ /* a bit flipped on the disk is put right there too */
				uint64_t blk = spill_base + (fo & ~(uint64_t)(BLK - 1)), before = cs.mended;
				size_t at = (fo & (BLK - 1)) + REC_HDR + 9;
				char good = tb[at];
				tb[at] ^= 0x10;
				if (pwrite(spill_fd, tb, 3 * BLK, blk) != 3 * BLK ||
				    nbd_io(fd, NBD_CMD_READ, c * BLK, tb, BLK) || memcmp(ta, tb, BLK) ||
				    cs.mended != before + 1)
					die("selftest: a flipped bit in the spill file was not put right");
				if (cs.rloc[cs.map[c]] == e && (pread(spill_fd, tb, 3 * BLK, blk) != 3 * BLK || tb[at] != good))
					die("selftest: a flipped bit in the spill file was read past but not mended");
				printf("spill file            a flipped bit is found and put right\n");
			}
			if (!err) { /* the same page written elsewhere must find its copy in the file */
				uint64_t d = c ? c - 1 : c + 1;
				char *keep = malloc(BLK);
				if (nbd_io(fd, NBD_CMD_READ, d * BLK, keep, BLK) ||
				    nbd_io(fd, NBD_CMD_WRITE, d * BLK, ta, BLK) || cs.map[d] != cs.map[c] ||
				    E_OFF(cs.rloc[cs.map[c]]) < phys_size ||
				    nbd_io(fd, NBD_CMD_READ, d * BLK, tb, BLK) || memcmp(ta, tb, BLK))
					die("selftest: a page was not matched with its copy in the spill file");
				if (nbd_io(fd, NBD_CMD_WRITE, d * BLK, keep, BLK))
					die("selftest: could not put a page back");
				free(keep);
				printf("spilled copies        are matched\n");
			}
			break;
		}
		t = now();
		check_all(fd, "after spilling");
		printf("incompressible    %llu MiB in VRAM, %llu MiB spilled%s; all read back at %.0f MiB/s\n",
		       (unsigned long long)((cs.bytes - cs.sbytes) / MiB), (unsigned long long)(cs.sbytes / MiB),
		       err ? ", then refused" : "", dev_size / MiB / (now() - t));
		for (int k = 1; k <= threads; k += threads - 1 ? threads - 1 : 1) {
			t = now();
			for (int i = 0; i < k; i++)
				pthread_create(&tc[i], NULL, rand_reader, (void *)(intptr_t)fds[i]);
			for (int i = 0; i < k; i++)
				pthread_join(tc[i], NULL);
			printf("4K random read    %7.0f IOPS  (%d thread%s, %llu%% of pages on disk)\n",
			       (double)RAND_OPS * k / (now() - t), k, k > 1 ? "s" : "",
			       (unsigned long long)(cs.sbytes * 100 / cs.bytes));
		}
		/* Page by page in the order written, as swap comes back in: mostly from the file's read-ahead. */
		uint64_t walk = npages / 4 < 60000 ? npages / 4 : 60000, hits0 = cs.held_hits;
		uint64_t dr0 = perf[P_DISK_READ].ops;
		t = now();
		for (uint64_t c = 0; c < walk; c++)
			check_pages(fd, c, 1, "reading through spilled pages");
		printf("4K in order       %7.0f IOPS  (1 thread; %llu disk reads for %llu pages)\n",
		       walk / (now() - t), (unsigned long long)(perf[P_DISK_READ].ops - dr0),
		       (unsigned long long)walk);
		(void)hits0;
		/* Free the newest half, which is what is in VRAM: the spilled pages should come back. */
		uint64_t half = npages / 2 & ~63ULL;
		for (uint64_t off = half * BLK; off < dev_size; off += 1024 * MiB) {
			uint64_t n = dev_size - off < 1024 * MiB ? dev_size - off : 1024 * MiB;
			if (nbd_io(fd, NBD_CMD_TRIM, off, NULL, n))
				die("selftest: discard failed");
		}
		memset(tver + half, 254, npages - half);
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

		/* The disk fails: writes must be refused once the card is full, not wait for ever. */
		spill_broken = 1;
		t = now();
		for (b = half, err = 0; b + 64 <= npages && !(err = put_pages(fd, b, 64, -1)); b += 64)
			;
		spill_broken = 0;
		if (err != ENOSPC && err != EIO)
			die("selftest: with the spill failing, filling the card gave error %d", err);
		check_all(fd, "after the spill failed and came back");
		if (put_pages(fd, 0, 64, 1))
			die("selftest: writes did not resume once the spill worked again");
		check_pages(fd, 0, 64, "after the spill came back");
		printf("failing spill     writes refused after %.1f s, nothing lost, and resumed when it came back\n",
		       now() - t);
	}

	for (uint64_t off = 0; off < dev_size; off += 1024 * MiB) {
		uint64_t n = dev_size - off < 1024 * MiB ? dev_size - off : 1024 * MiB;
		if (nbd_io(fd, NBD_CMD_TRIM, off, NULL, n))
			die("selftest: discard failed");
	}
	memset(tver, 0, npages);
	memset(ta, 0, BLK);
	if (cs.pages || cs.recs || cs.bytes || nbd_io(fd, NBD_CMD_READ, (npages / 2) * BLK, tb, BLK) || memcmp(ta, tb, BLK))
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
		else if (!strcmp(argv[i], "--cipher") && v)
			cipher_name = v, i++;
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
	if (cratio && !(cratio >= 1 && cratio <= 8))
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
	/*
	 * Our memory holds other programs' pages and the keys. It is locked, so
	 * it is never swapped; this makes sure it is never written to a core
	 * file either, and that only root can look at it through /proc. Nothing
	 * keeps root out, who can read the device itself anyway.
	 */
	struct rlimit none = { 0, 0 };
	if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) || setrlimit(RLIMIT_CORE, &none))
		fprintf(stderr, "vroom: warning: cannot rule out core dumps: %s\n", strerror(errno));
	if (dev_path && !do_bench) { /* figures left by an earlier daemon are not ours */
		char path[128];
		const char *base = strrchr(dev_path, '/');
		snprintf(path, sizeof path, "/run/vroom/%s.stat", base ? base + 1 : dev_path);
		unlink(path);
		snprintf(path, sizeof path, "/run/vroom/%s.count", base ? base + 1 : dev_path);
		unlink(path);
		snprintf(path, sizeof path, "/run/vroom/%s.pid", base ? base + 1 : dev_path);
		unlink(path);
		mkdir("/run/vroom", 0755);
	}
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
	if (getenv("VROOM_CIPHER") && *getenv("VROOM_CIPHER"))
		cipher_name = getenv("VROOM_CIPHER");
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
	if (!test) {
		char path[128];
		const char *base = strrchr(dev_path, '/');
		snprintf(path, sizeof path, "/run/vroom/%s.pid", base ? base + 1 : dev_path);
		unlink(path);
		/*
		 * A compressed store leaves only noise behind. A plain one has
		 * other programs' pages on the card as they were, and the next
		 * program to be given that memory should not find them there.
		 */
		for (uint64_t i = 0; !cratio && i < (phys_size + CHUNK - 1) / CHUNK; i++)
			cu.MemsetD8(chunks[i], 0, phys_size - i * CHUNK < CHUNK ? phys_size - i * CHUNK : CHUNK);
	}
	if (test) {
		static const char *what[P_KINDS] = { "requests read", "requests write", "VRAM read", "VRAM write",
						     "disk read", "disk write" };
		static const struct perf none;
		printf("\ntimed 1 in %d      operations   avg size       p50       p95       p99\n", SAMPLE);
		for (int k = 0; k < P_KINDS; k++)
			if (perf[k].t_ops)
				printf("%-15s %12llu %8.0f KiB %7.1f us %7.1f us %7.1f us\n", what[k],
				       (unsigned long long)perf[k].ops, perf[k].bytes / 1024.0 / perf[k].ops,
				       hist_quantile(&perf[k], &none, 0.50) / 1e3,
				       hist_quantile(&perf[k], &none, 0.95) / 1e3,
				       hist_quantile(&perf[k], &none, 0.99) / 1e3);
	}
	if (counting && test)
		count_print(stdout);
	return 0;
}
