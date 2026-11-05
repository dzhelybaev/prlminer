// ejjjminer — Pearl stratum client (object-params dialect used by Suprnova, LuckyPool,
// HeroMiners, Kryptex, ...):
//   -> mining.authorize {wallet, worker, pass, agent}
//   <- mining.notify    {job_id, header (76 bytes hex), target (u256 BE hex), height}
//   -> mining.submit    {job_id, plain_proof (base64 PlainProof)}
// The only network connection ejjjminer makes is this one, to the pool you configure.
#include "stratum.h"
#include "json.h"
#include "log.h"
#include "miner.h"
#include "gz.h"

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static pool_ep g_pools[MAX_POOLS];
static int g_npools, g_cur_pool;
static char g_wallet[256], g_worker[128], g_pass[256];
static int g_sock = -1;
static pthread_mutex_t g_send_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_th;
static volatile int g_quit, g_connected;
static uint64_t g_gen;
static int g_next_id = 10;
static double g_ping_ms;
static volatile int g_gzip;   // pool accepted "type":"v2" (gzip-compressed proofs)

#define PEND 512
static struct { int id, dev; double t; } g_pend[PEND];
static pthread_mutex_t g_pend_mu = PTHREAD_MUTEX_INITIALIZER;

int stratum_parse_url(const char *url, pool_ep *out) {
    memset(out, 0, sizeof(*out));
    const char *p = strstr(url, "://");
    if (p) {
        if (strncmp(url, "stratum+tcp", 11) && strncmp(url, "stratum", 7) && strncmp(url, "tcp", 3)) {
            LOGE("pool URL scheme not supported (use a plain TCP port): %s", url);
            return -1;
        }
        if (!strncmp(url, "stratum+ssl", 11) || !strncmp(url, "stratum+tls", 11) || !strncmp(url, "ssl", 3)) {
            LOGE("TLS pool ports are not supported, use the pool's plain TCP port: %s", url);
            return -1;
        }
        url = p + 3;
    }
    const char *c = strrchr(url, ':');
    if (!c) return -1;
    size_t hl = (size_t)(c - url);
    if (hl == 0 || hl >= sizeof(out->host)) return -1;
    memcpy(out->host, url, hl);
    out->port = atoi(c + 1);
    return out->port > 0 && out->port < 65536 ? 0 : -1;
}

void stratum_configure(const pool_ep *pools, int n, const char *wallet, const char *worker, const char *pass) {
    g_npools = n > MAX_POOLS ? MAX_POOLS : n;
    memcpy(g_pools, pools, sizeof(pool_ep) * g_npools);
    snprintf(g_wallet, sizeof g_wallet, "%s", wallet);
    snprintf(g_worker, sizeof g_worker, "%s", worker ? worker : "");
    snprintf(g_pass, sizeof g_pass, "%s", pass ? pass : "x");
}

uint64_t stratum_conn_gen(void) { return __atomic_load_n(&g_gen, __ATOMIC_ACQUIRE); }
int stratum_connected(void) { return g_connected; }
double stratum_ping_ms(void) { return g_ping_ms; }
const char *stratum_pool(void) {
    static char buf[256];
    if (!g_npools) return "";
    snprintf(buf, sizeof buf, "%s:%d", g_pools[g_cur_pool].host, g_pools[g_cur_pool].port);
    return buf;
}

static int send_all(const char *s, size_t n) {
    pthread_mutex_lock(&g_send_mu);
    int fd = g_sock, rc = 0;
    while (n && fd >= 0) {
        ssize_t w = send(fd, s, n, MSG_NOSIGNAL);
        if (w < 0) { if (errno == EINTR) continue; rc = -1; break; }
        s += w; n -= (size_t)w;
    }
    if (fd < 0) rc = -1;
    pthread_mutex_unlock(&g_send_mu);
    return rc;
}

static int connect_to(const pool_ep *ep) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char port[16];
    snprintf(port, sizeof port, "%d", ep->port);
    int e = getaddrinfo(ep->host, port, &hints, &res);
    if (e) { LOGW("DNS lookup failed for %s: %s", ep->host, gai_strerror(e)); return -1; }
    int fd = -1;
    for (struct addrinfo *a = res; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv = {15, 0};
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef TCP_KEEPIDLE
    int idle = 60, intv = 15, cnt = 4;
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof idle);
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intv, sizeof intv);
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof cnt);
#endif
    return fd;
}

static int hex_to_bytes(const char *h, uint8_t *out, size_t n) {
    if (strlen(h) != 2 * n) return -1;
    for (size_t i = 0; i < n; i++) {
        unsigned v;
        if (sscanf(h + 2 * i, "%2x", &v) != 1) return -1;
        out[i] = (uint8_t)v;
    }
    return 0;
}

static void parse_error(const jval *err, int *code, char *msg, size_t ml) {
    *code = -1;
    msg[0] = 0;
    double d;
    if (!err || err->t == J_NULL) return;
    if (err->t == J_OBJ) {
        if (json_num(json_get(err, "code"), &d)) *code = (int)d;
        const char *m = json_str(json_get(err, "msg"));
        if (!m) m = json_str(json_get(err, "message"));
        if (m) snprintf(msg, ml, "%s", m);
    } else if (err->t == J_ARR && err->n >= 1) {
        if (json_num(&err->kids[0], &d)) *code = (int)d;
        if (err->n >= 2 && json_str(&err->kids[1])) snprintf(msg, ml, "%s", err->kids[1].str);
    } else if (err->t == J_STR) {
        snprintf(msg, ml, "%s", err->str);
    }
}

static void handle_notify(const jval *params, uint64_t gen) {
    if (!params || params->t != J_OBJ) {
        LOGW("mining.notify with unsupported params format (this pool uses another stratum dialect)");
        return;
    }
    const char *id = json_str(json_get(params, "job_id"));
    const char *hdr = json_str(json_get(params, "header"));
    const char *tgt = json_str(json_get(params, "target"));
    double h = 0;
    json_num(json_get(params, "height"), &h);
    if (!id || !hdr || !tgt) { LOGW("mining.notify missing fields"); return; }
    uint8_t header[PRL_HEADER_LEN];
    if (hex_to_bytes(hdr, header, PRL_HEADER_LEN)) { LOGW("bad header in mining.notify (%zu hex chars)", strlen(hdr)); return; }
    u256 target;
    u256_from_be_hex(&target, tgt);
    miner_on_notify(id, header, &target, (uint32_t)h, gen);
}

static void handle_line(const char *line, uint64_t gen) {
    jval *v = json_parse(line);
    if (!v || v->t != J_OBJ) { LOGW("pool sent invalid JSON: %.120s", line); json_free(v); return; }
    const char *method = json_str(json_get(v, "method"));
    if (method) {
        if (!strcmp(method, "mining.notify")) handle_notify(json_get(v, "params"), gen);
        else if (!strcmp(method, "client.show_message") || !strcmp(method, "mining.message"))
            LOGI("pool message: %.200s", line);
        else LOGD("ignoring pool method %s", method);
        json_free(v);
        return;
    }
    double idd;
    if (!json_num(json_get(v, "id"), &idd)) { json_free(v); return; }
    int id = (int)idd;
    const jval *res = json_get(v, "result"), *err = json_get(v, "error");
    int code;
    char msg[256];
    parse_error(err, &code, msg, sizeof msg);
    bool ok = res && ((res->t == J_BOOL && res->b) || res->t == J_OBJ || (res->t == J_STR)) && (!err || err->t == J_NULL);
    if (id == 1) {
        const char *ty = json_str(json_get(v, "type"));
        if (!ty && res && res->t == J_OBJ) ty = json_str(json_get(res, "type"));
        g_gzip = ty && !strcmp(ty, "v2");
        if (ok) LOGI("authorized on %s as %s%s%s%s", stratum_pool(), g_wallet, g_worker[0] ? "." : "", g_worker,
                     g_gzip ? " (gzip proofs)" : "");
        else LOGE("authorization failed: code %d %s", code, msg);
        json_free(v);
        return;
    }
    int dev = -1;
    double t = 0;
    pthread_mutex_lock(&g_pend_mu);
    for (int i = 0; i < PEND; i++)
        if (g_pend[i].id == id) { dev = g_pend[i].dev; t = g_pend[i].t; g_pend[i].id = 0; break; }
    pthread_mutex_unlock(&g_pend_mu);
    if (dev >= 0 && !ok) LOGW("pool reply to share id %d: %.300s", id, line);
    if (dev >= 0) {
        g_ping_ms = (now_sec() - t) * 1e3;
        miner_on_share_result(dev, ok, code, msg);
    }
    json_free(v);
}

static void *stratum_main(void *arg) {
    (void)arg;
    int fails = 0;
    size_t cap = 1 << 16;
    char *buf = malloc(cap);
    while (!g_quit) {
        const pool_ep *ep = &g_pools[g_cur_pool];
        LOGI("connecting to %s:%d", ep->host, ep->port);
        int fd = connect_to(ep);
        if (fd < 0) {
            fails++;
            int wait = fails < 6 ? (1 << fails) : 60;
            LOGW("connection to %s:%d failed, retry in %d s", ep->host, ep->port, wait);
            if (fails % 3 == 0 && g_npools > 1) {
                g_cur_pool = (g_cur_pool + 1) % g_npools;
                LOGI("switching to pool %s", stratum_pool());
            }
            for (int i = 0; i < wait * 10 && !g_quit; i++) usleep(100000);
            continue;
        }
        pthread_mutex_lock(&g_send_mu);
        g_sock = fd;
        pthread_mutex_unlock(&g_send_mu);
        uint64_t gen = __atomic_add_fetch(&g_gen, 1, __ATOMIC_RELEASE);
        g_connected = 1;
        g_gzip = 0;
        snprintf(g_miner.pool_url, sizeof g_miner.pool_url, "%s", stratum_pool());
        // authorize (login may be "wallet.worker")
        char wallet[256], worker[128];
        snprintf(wallet, sizeof wallet, "%s", g_wallet);
        snprintf(worker, sizeof worker, "%s", g_worker);
        // The login is sent verbatim (pools split "wallet.worker" themselves); the worker
        // field carries the worker name explicitly for pools that read it separately.
        char *dot = strchr(wallet, '.');
        if (dot && !worker[0]) snprintf(worker, sizeof worker, "%s", dot + 1);
        if (!worker[0]) snprintf(worker, sizeof worker, "default");
        char ew[600], ek[300], ep2[600], msg[2048];
        json_escape(ew, sizeof ew, wallet);
        json_escape(ek, sizeof ek, worker);
        json_escape(ep2, sizeof ep2, g_pass);
        snprintf(msg, sizeof msg,
                 "{\"id\":1,\"jsonrpc\":\"2.0\",\"method\":\"mining.authorize\",\"params\":{\"wallet\":%s,\"worker\":%s,"
                 "\"pass\":%s,\"agent\":\"" PRL_AGENT "\",\"type\":\"v2\"}}\n",
                 ew, ek, ep2);
        send_all(msg, strlen(msg));
        size_t len = 0;
        double last_rx = now_sec();
        int alive = 1;
        while (!g_quit && alive) {
            struct pollfd pfd = {fd, POLLIN, 0};
            int pr = poll(&pfd, 1, 500);
            if (pr < 0 && errno != EINTR) break;
            {   // warn about shares the pool never answered
                double t = now_sec();
                pthread_mutex_lock(&g_pend_mu);
                for (int i = 0; i < PEND; i++)
                    if (g_pend[i].id && t - g_pend[i].t > 60) {
                        LOGW("pool did not answer share id %d within 60 s", g_pend[i].id);
                        g_pend[i].id = 0;
                    }
                pthread_mutex_unlock(&g_pend_mu);
            }
            if (pr <= 0) {
                if (now_sec() - last_rx > 600) { LOGW("no data from pool for 10 minutes, reconnecting"); break; }
                continue;
            }
            if (len + 65536 + 1 > cap) {
                if (cap >= (32u << 20)) { LOGW("pool line too long"); break; }
                cap *= 2;
                buf = realloc(buf, cap);
            }
            ssize_t r = recv(fd, buf + len, cap - len - 1, 0);
            if (r <= 0) { if (r < 0 && errno == EINTR) continue; LOGW("pool closed the connection"); break; }
            last_rx = now_sec();
            len += (size_t)r;
            buf[len] = 0;
            char *s = buf, *nl;
            while ((nl = memchr(s, '\n', len - (size_t)(s - buf)))) {
                *nl = 0;
                if (nl > s && nl[-1] == '\r') nl[-1] = 0;
                if (*s) handle_line(s, gen);
                s = nl + 1;
            }
            len -= (size_t)(s - buf);
            memmove(buf, s, len);
            fails = 0;
        }
        g_connected = 0;
        pthread_mutex_lock(&g_send_mu);
        close(fd);
        g_sock = -1;
        pthread_mutex_unlock(&g_send_mu);
        miner_on_disconnect();
        if (!g_quit) {
            fails++;
            for (int i = 0; i < 30 && !g_quit; i++) usleep(100000);
            if (fails % 3 == 0 && g_npools > 1) g_cur_pool = (g_cur_pool + 1) % g_npools;
        }
    }
    free(buf);
    return NULL;
}

int stratum_submit(const char *job_id, const uint8_t *proof, size_t plen, int dev) {
    char eid[600];
    json_escape(eid, sizeof eid, job_id);
    uint8_t *gz = NULL;
    size_t glen = 0;
    if (g_gzip) gz = prl_gzip(proof, plen, &glen);
    char *b64 = gz ? prl_base64(gz, glen) : prl_base64(proof, plen);
    free(gz);
    if (!b64) return -1;
    int id;
    pthread_mutex_lock(&g_pend_mu);
    id = ++g_next_id;
    if (g_next_id > 1000000000) g_next_id = 10;
    int slot = id % PEND;
    g_pend[slot].id = id;
    g_pend[slot].dev = dev;
    g_pend[slot].t = now_sec();
    pthread_mutex_unlock(&g_pend_mu);
    size_t bl = strlen(b64), cap = bl + 1024;
    char *m = malloc(cap);
    if (!m) { free(b64); return -1; }
    int n = snprintf(m, cap, "{\"id\":%d,\"jsonrpc\":\"2.0\",\"method\":\"mining.submit\",\"params\":{\"job_id\":%s,\"plain_proof\":\"",
                     id, eid);
    memcpy(m + n, b64, bl);
    n += (int)bl;
    n += snprintf(m + n, cap - n, "\"}}\n");
    free(b64);
    int rc = send_all(m, (size_t)n);
    free(m);
    if (rc) LOGW("failed to send share to pool");
    else if (g_gzip) LOGI("share sent: id %d, proof %zu bytes, gzip %zu bytes", id, plen, glen);
    else LOGI("share sent: id %d, proof %zu bytes (uncompressed)", id, plen);
    return rc;
}

int stratum_start(void) {
    if (!g_npools) return -1;
    return pthread_create(&g_th, NULL, stratum_main, NULL);
}

void stratum_stop(void) {
    g_quit = 1;
    pthread_mutex_lock(&g_send_mu);
    if (g_sock >= 0) shutdown(g_sock, SHUT_RDWR);
    pthread_mutex_unlock(&g_send_mu);
    pthread_join(g_th, NULL);
}
