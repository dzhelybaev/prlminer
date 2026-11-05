#pragma once
#include <stdint.h>

enum { LOG_DEBUG = 0, LOG_INFO = 1, LOG_WARN = 2, LOG_ERR = 3 };
extern int g_log_level;
extern int g_log_color;
void log_open_file(const char *path);
void logmsg(int level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#define LOGD(...) logmsg(LOG_DEBUG, __VA_ARGS__)
#define LOGI(...) logmsg(LOG_INFO, __VA_ARGS__)
#define LOGW(...) logmsg(LOG_WARN, __VA_ARGS__)
#define LOGE(...) logmsg(LOG_ERR, __VA_ARGS__)

double now_sec(void);          // monotonic seconds
uint64_t wall_ms(void);
const char *fmt_rate(double v, char *buf, int n);   // "12.34 T"
