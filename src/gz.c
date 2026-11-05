// gzip compression of PlainProofs (Kryptex "type":"v2" stratum extension).
// Uses the system zlib (libz.so.1, loaded at run time) with wbits=31 (gzip container);
// falls back to an uncompressed but valid gzip stream if zlib is unavailable.
#include "gz.h"
#include "log.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

typedef int (*deflateInit2_fn)(z_streamp, int, int, int, int, int, const char *, int);
typedef int (*deflate_fn)(z_streamp, int);
typedef int (*deflateEnd_fn)(z_streamp);
typedef uLong (*deflateBound_fn)(z_streamp, uLong);

static struct { int tried, ok; deflateInit2_fn init; deflate_fn def; deflateEnd_fn end; deflateBound_fn bound; } Z;

static void zload(void) {
    if (Z.tried) return;
    Z.tried = 1;
    void *h = dlopen("libz.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) { LOGW("libz.so.1 not found: shares will be sent as uncompressed gzip"); return; }
    *(void **)&Z.init = dlsym(h, "deflateInit2_");
    *(void **)&Z.def = dlsym(h, "deflate");
    *(void **)&Z.end = dlsym(h, "deflateEnd");
    *(void **)&Z.bound = dlsym(h, "deflateBound");
    Z.ok = Z.init && Z.def && Z.end && Z.bound;
}

static uint32_t crc32_ieee(const uint8_t *p, size_t n) {
    static uint32_t T[256];
    static int init;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            T[i] = c;
        }
        init = 1;
    }
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; i++) c = T[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xffffffffu;
}

static uint8_t *gzip_stored(const uint8_t *in, size_t n, size_t *outlen) {
    size_t blocks = n / 65535 + 1, cap = 18 + n + 5 * blocks;
    uint8_t *o = malloc(cap), *p = o;
    if (!o) return NULL;
    const uint8_t hdr[10] = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff};
    memcpy(p, hdr, 10);
    p += 10;
    size_t off = 0;
    do {
        size_t len = n - off > 65535 ? 65535 : n - off;
        *p++ = (off + len == n) ? 1 : 0;
        p[0] = len & 0xff; p[1] = len >> 8; p[2] = ~len & 0xff; p[3] = (~len >> 8) & 0xff;
        p += 4;
        memcpy(p, in + off, len);
        p += len;
        off += len;
    } while (off < n);
    uint32_t crc = crc32_ieee(in, n), sz = (uint32_t)n;
    for (int i = 0; i < 4; i++) *p++ = (uint8_t)(crc >> (8 * i));
    for (int i = 0; i < 4; i++) *p++ = (uint8_t)(sz >> (8 * i));
    *outlen = (size_t)(p - o);
    return o;
}

uint8_t *prl_gzip(const uint8_t *in, size_t n, size_t *outlen) {
    zload();
    if (!Z.ok) return gzip_stored(in, n, outlen);
    z_stream s;
    memset(&s, 0, sizeof s);
    if (Z.init(&s, 9, Z_DEFLATED, 31, 8, Z_DEFAULT_STRATEGY, ZLIB_VERSION, (int)sizeof(z_stream)) != Z_OK)
        return gzip_stored(in, n, outlen);
    size_t cap = Z.bound(&s, (uLong)n) + 64;
    uint8_t *o = malloc(cap);
    if (!o) { Z.end(&s); return NULL; }
    s.next_in = (Bytef *)in;
    s.avail_in = (uInt)n;
    s.next_out = o;
    s.avail_out = (uInt)cap;
    int r = Z.def(&s, Z_FINISH);
    *outlen = s.total_out;
    Z.end(&s);
    if (r != Z_STREAM_END) { free(o); return gzip_stored(in, n, outlen); }
    return o;
}
