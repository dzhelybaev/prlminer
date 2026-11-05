#include "log.h"
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int g_log_level = LOG_INFO;
int g_log_color = -1;
static FILE *g_logf;
static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;

void log_open_file(const char *path) {
    g_logf = fopen(path, "a");
    if (g_logf) setvbuf(g_logf, NULL, _IOLBF, 0);
}

void logmsg(int level, const char *fmt, ...) {
    if (level < g_log_level) return;
    if (g_log_color < 0) g_log_color = isatty(1);
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    char ts[32];
    strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", &tm);
    static const char *tag[] = {"DBG", "INF", "WRN", "ERR"};
    static const char *col[] = {"\033[90m", "", "\033[33m", "\033[31m"};
    pthread_mutex_lock(&g_log_mu);
    if (g_log_color) printf("%s[%s] %s %s\033[0m\n", col[level], ts, tag[level], msg);
    else printf("[%s] %s %s\n", ts, tag[level], msg);
    fflush(stdout);
    if (g_logf) fprintf(g_logf, "[%s] %s %s\n", ts, tag[level], msg);
    pthread_mutex_unlock(&g_log_mu);
}

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

uint64_t wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

const char *fmt_rate(double v, char *buf, int n) {
    const char *u[] = {"", "K", "M", "G", "T", "P", "E"};
    int i = 0;
    while (v >= 1000 && i < 6) { v /= 1000; i++; }
    snprintf(buf, n, "%.2f %s", v, u[i]);
    return buf;
}
