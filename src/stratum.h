#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { char host[192]; int port; } pool_ep;

int stratum_parse_url(const char *url, pool_ep *out);   // 0 ok
void stratum_configure(const pool_ep *pools, int n, const char *wallet, const char *worker, const char *pass);
int stratum_start(void);
void stratum_stop(void);
uint64_t stratum_conn_gen(void);       // increments on every (re)connect; 0 = never connected
int stratum_connected(void);
int stratum_submit(const char *job_id, const uint8_t *proof, size_t len, int dev);   // gzip if negotiated
const char *stratum_pool(void);
double stratum_ping_ms(void);
