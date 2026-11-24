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
 *   vroom --nbd /dev/nbd0 [--size MiB]     serve the device (root)
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
	dev_size = size_mib * MiB;
	fprintf(stderr, "vroom: %s: %llu MiB free of %llu MiB, taking %llu MiB\n", name,
		free_b / MiB, total_b / MiB, (unsigned long long)size_mib);

	uint64_t n = (dev_size + CHUNK - 1) / CHUNK;
	chunks = calloc(n, sizeof *chunks);
	for (uint64_t i = 0; i < n; i++) {
		uint64_t len = dev_size - i * CHUNK < CHUNK ? dev_size - i * CHUNK : CHUNK;
		if ((r = cu.MemAlloc(&chunks[i], len)))
			die("VRAM allocation failed at %llu MiB (CUDA error %d)",
			    (unsigned long long)(i * CHUNK / MiB), r);
		if ((r = cu.MemsetD8(chunks[i], 0, len)))
			die("VRAM clear failed (CUDA error %d)", r);
	}
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
			if (!err && vram_rw(0, off, buf, n))
				err = EIO;
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
				if (vram_rw(0, off, buf, n))
					goto out; /* mid-reply: drop the connection */
			}
			continue;
		case NBD_CMD_WRITE:
			while (len) {
				n = len < BUFSZ ? len : BUFSZ;
				if (rd_full(fd, buf, n))
					goto out;
				if (!err && vram_rw(1, off, buf, n))
					err = EIO;
				off += n, len -= n;
			}
			break;
		case NBD_CMD_FLUSH:
			err = 0;
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
static const char *dev_path;
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
		  (unsigned long)(NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH | NBD_FLAG_CAN_MULTI_CONN)))
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
				else if (vram_rw((d->op_flags & 0xff) == UBLK_IO_OP_WRITE, off,
						 buf[tag], len))
					res = -EIO;
				else
					res = len;
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
					    .dev_sectors = dev_size >> 9 } };
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
		else
			die("usage: vroom (--selftest | --dev /dev/nbdN | --dev /dev/ublkbN | --bench DEV)"
			    " [--size MiB] [--gpu N] [--threads N]");
	}
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
	if (test)
		selftest(threads);
	else if (ublk)
		serve_ublk(threads);
	else
		serve_nbd(threads);
	return 0;
}
