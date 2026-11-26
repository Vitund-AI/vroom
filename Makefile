VERSION = 0.5.0
PREFIX ?= /usr/local
SBINDIR = $(PREFIX)/sbin
UNITDIR = $(PREFIX)/lib/systemd/system
CFLAGS ?= -O2 -Wall -Wextra

ARCH = $(shell dpkg --print-architecture)
DEBROOT = build/deb
DEB = vroom_$(VERSION)_$(ARCH).deb

all: vroom vroom@.service

vroom: vroom.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< -lpthread -ldl

vroom@.service: vroom@.service.in FORCE
	sed 's|@SBINDIR@|$(SBINDIR)|g' $< > $@

install: all
	install -Dm755 vroom $(DESTDIR)$(SBINDIR)/vroom
	install -Dm755 vroomctl $(DESTDIR)$(SBINDIR)/vroomctl
	install -Dm644 vroom@.service $(DESTDIR)$(UNITDIR)/vroom@.service
	install -d $(DESTDIR)/etc/vroom

uninstall:
	rm -f $(DESTDIR)$(SBINDIR)/vroom $(DESTDIR)$(SBINDIR)/vroomctl $(DESTDIR)$(UNITDIR)/vroom@.service

# Binary .deb built straight with dpkg-deb; needs no debhelper.
deb:
	rm -rf $(DEBROOT)
	$(MAKE) install PREFIX=/usr DESTDIR=$(DEBROOT)
	install -Dm644 README.md $(DEBROOT)/usr/share/doc/vroom/README.md
	install -Dm644 LICENSE $(DEBROOT)/usr/share/doc/vroom/copyright
	install -Dm644 packaging/needrestart.conf $(DEBROOT)/etc/needrestart/conf.d/vroom.conf
	install -d $(DEBROOT)/DEBIAN
	sed -e 's/@VERSION@/$(VERSION)/' -e 's/@ARCH@/$(ARCH)/' packaging/control.in > $(DEBROOT)/DEBIAN/control
	install -m755 packaging/postinst packaging/prerm packaging/postrm $(DEBROOT)/DEBIAN/
	dpkg-deb --root-owner-group --build $(DEBROOT) $(DEB)

# The self-tests against a card made of ordinary memory, under the address and
# undefined-behaviour sanitizers: needs no GPU and no root. (The spill file is
# opened O_DIRECT, so build/ must not be on tmpfs.)
TEST = ASAN_OPTIONS=detect_leaks=0 timeout 600 build/vroom-test --selftest
build/vroom-test: vroom.c
	mkdir -p build
	$(CC) -O1 -g -Wall -Wextra -DFAKE_CARD -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ $< -lpthread -ldl

check: build/vroom-test
	bash -n vroomctl
	$(TEST) --size 64 > build/check.log 2>&1 || { tail -20 build/check.log; exit 1; }
	$(TEST) --size 64 --compress 1.5 >> build/check.log 2>&1 || { tail -20 build/check.log; exit 1; }
	$(TEST) --size 64 --compress 2.2 --spill build/check.spill >> build/check.log 2>&1 || { tail -20 build/check.log; exit 1; }
	$(TEST) --size 48 --compress 3 --spill build/check.spill --spill-size 40 --threads 8 --cipher chacha20 >> build/check.log 2>&1 || { tail -20 build/check.log; exit 1; }
	@grep -c 'selftest passed' build/check.log | sed 's/$$/ self-tests passed (build\/check.log)/'

clean:
	rm -rf vroom vroom@.service build *.deb

FORCE:
.PHONY: all install uninstall deb check clean FORCE
