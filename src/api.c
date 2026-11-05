// ejjjminer — local read-only HTTP/JSON stats API (for HiveOS h-stats.sh). Binds 127.0.0.1 by default.
#include "api.h"
#include "log.h"
#include "miner.h"
#include "stratum.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int g_lfd = -1;
static pthread_t g_api_th;

int api_render(char *out, size_t cap) {
    miner_t *M = &g_miner;
    size_t n = 0;
    double total = 0;
    for (int i = 0; i < M->ndev; i++) total += M->dev[i].rate;
    n += snprintf(out + n, cap - n,
                  "{\"miner\":\"ejjjminer\",\"version\":\"%s\",\"algo\":\"pearlhash\",\"uptime\":%.0f,"
                  "\"pool\":\"%s\",\"connected\":%s,\"height\":%u,\"difficulty\":%.6g,\"hashrate\":%.0f,"
                  "\"accepted\":%u,\"rejected\":%u,\"stale\":%u,\"invalid\":%u,\"gpus\":[",
                  PRL_VERSION, now_sec() - M->start_t, M->pool_url, stratum_connected() ? "true" : "false",
                  M->height, M->diff, total, atomic_load(&M->shares_accepted), atomic_load(&M->shares_rejected),
                  atomic_load(&M->shares_stale), atomic_load(&M->shares_invalid));
    for (int i = 0; i < M->ndev && n < cap; i++) {
        dev_stats *d = &M->dev[i];
        unsigned bus = 0;
        sscanf(d->busid, "%x", &bus);
        n += snprintf(out + n, cap - n,
                      "%s{\"id\":%d,\"name\":\"%s\",\"busid\":\"%s\",\"bus\":%u,\"kernel\":\"%s\",\"hashrate\":%.0f,"
                      "\"accepted\":%u,\"rejected\":%u,\"stale\":%u,\"invalid\":%u,\"active\":%s}",
                      i ? "," : "", i, d->name, d->busid, bus, d->kernel, d->rate, atomic_load(&d->accepted),
                      atomic_load(&d->rejected), atomic_load(&d->stale), atomic_load(&d->invalid),
                      d->active ? "true" : "false");
    }
    if (n < cap) n += snprintf(out + n, cap - n, "]}");
    return (int)(n < cap ? n : cap - 1);
}

static void *api_main(void *arg) {
    (void)arg;
    char req[2048], body[16384], resp[17000];
    for (;;) {
        int c = accept(g_lfd, NULL, NULL);
        if (c < 0) { if (errno == EINTR) continue; if (g_miner.quit) break; usleep(100000); continue; }
        struct timeval tv = {2, 0};
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        ssize_t r = recv(c, req, sizeof req - 1, 0);
        (void)r;
        int bl = api_render(body, sizeof body);
        int rl = snprintf(resp, sizeof resp,
                          "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n"
                          "Connection: close\r\n\r\n%s", bl, body);
        send(c, resp, (size_t)rl, MSG_NOSIGNAL);
        close(c);
    }
    return NULL;
}

int api_start(const char *host, int port) {
    g_lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_lfd < 0) return -1;
    int one = 1;
    setsockopt(g_lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(g_lfd, (struct sockaddr *)&a, sizeof a) || listen(g_lfd, 8)) {
        LOGW("API: cannot listen on %s:%d (%s)", host, port, strerror(errno));
        close(g_lfd);
        g_lfd = -1;
        return -1;
    }
    LOGI("API listening on http://%s:%d", host, port);
    return pthread_create(&g_api_th, NULL, api_main, NULL);
}
