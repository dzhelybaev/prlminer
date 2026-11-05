// ejjjminer — open-source Pearl (PRL) GPU miner for NVIDIA. No dev fee, no telemetry.
#include "api.h"
#include "backend.h"
#include "log.h"
#include "miner.h"
#include "stratum.h"

#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop;
static void on_signal(int s) { (void)s; g_stop = 1; }

static void usage(void) {
    printf(
        "ejjjminer %s — Pearl (PRL, pearlhash) miner for NVIDIA GPUs. No dev fee, no telemetry.\n\n"
        "usage: ejjjminer -o HOST:PORT -u WALLET[.WORKER] [options]\n\n"
        "pool:\n"
        "  -o, --pool URL          stratum pool (stratum+tcp://host:port or host:port); repeat for failover\n"
        "  -u, --user LOGIN        wallet address, optionally WALLET.WORKER\n"
        "  -w, --worker NAME       worker name (default: taken from LOGIN or 'default')\n"
        "  -p, --pass PASS         pool password (default x; e.g. d=1000000 for fixed difficulty)\n"
        "devices:\n"
        "  -d, --devices LIST      CUDA device indices, e.g. 0,1,3 (default: all)\n"
        "      --kernel K          auto | mma (tensor cores, RTX 20xx+) | dp4a (default auto)\n"
        "      --fused M           auto | on | off: noise generated inside the GEMM kernel\n"
        "                          (auto = measure all kernels at start, use the fastest; default auto)\n"
        "      --variant V         force a kernel variant: fmma | v2a | v2b | v2c (default: fastest)\n"
        "      --rows N            matrix rows per attempt (default: 128 x SM count)\n"
        "      --list-devices      list GPUs and exit\n"
        "      --selftest          run the GPU self-test and exit\n"
        "      --cpu               mine on the CPU reference backend (testing only, very slow)\n"
        "protocol:\n"
        "      --tile W            hash tile width / 2: 32 (official 2x64, default), 16, 8\n"
        "      --k K               common dimension (default 8192)\n"
        "      --cols N            matrix columns (default 2048)\n"
        "      --cert C            auto | 2 | 3: noise-seed derivation (auto: v3 from height %d)\n"
        "      --salted-height H   fork height for v3 seeds in auto mode\n"
        "      --share-target T    scaled (target x tile work, default) | raw\n"
        "misc:\n"
        "      --api-port P        stats API port on 127.0.0.1 (default 4068, 0 = off)\n"
        "      --api-host H        stats API bind address (default 127.0.0.1)\n"
        "      --benchmark SEC     mine a synthetic job for SEC seconds (no pool)\n"
        "      --bench-diff D      share difficulty for the benchmark (default 100000)\n"
        "      --print-interval S  summary interval in seconds (default 30)\n"
        "      --log FILE          also write the log to FILE\n"
        "      --debug             verbose logging\n"
        "  -h, --help\n",
        PRL_VERSION, 99000);
}

enum {
    O_KERNEL = 1000, O_FUSED, O_VARIANT, O_ROWS, O_LIST, O_SELFTEST, O_CPU, O_TILE, O_K, O_COLS, O_CERT, O_SALTH, O_APIPORT,
    O_APIHOST, O_BENCH, O_BENCHDIFF, O_PRINT, O_LOG, O_DEBUG, O_SHARET
};

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    pool_ep pools[MAX_POOLS];
    int npools = 0;
    const char *user = NULL, *worker = NULL, *pass = "x", *devices = NULL, *kernel = "auto", *cert = "auto";
    const char *api_host = "127.0.0.1", *logfile = NULL;
    int rows = 0, tile = 32, k = 8192, cols = 2048, api_port = 4068, list = 0, selftest = 0, use_cpu = 0;
    int salted_h = 99000, bench = 0, print_iv = 30;
    double bench_diff = 100000;
    static struct option opts[] = {
        {"pool", 1, 0, 'o'}, {"url", 1, 0, 'o'}, {"user", 1, 0, 'u'}, {"wallet", 1, 0, 'u'}, {"worker", 1, 0, 'w'},
        {"pass", 1, 0, 'p'}, {"devices", 1, 0, 'd'}, {"kernel", 1, 0, O_KERNEL}, {"fused", 1, 0, O_FUSED}, {"variant", 1, 0, O_VARIANT}, {"rows", 1, 0, O_ROWS},
        {"list-devices", 0, 0, O_LIST}, {"selftest", 0, 0, O_SELFTEST}, {"cpu", 0, 0, O_CPU},
        {"tile", 1, 0, O_TILE}, {"k", 1, 0, O_K}, {"cols", 1, 0, O_COLS}, {"cert", 1, 0, O_CERT},
        {"salted-height", 1, 0, O_SALTH}, {"api-port", 1, 0, O_APIPORT}, {"api-host", 1, 0, O_APIHOST},
        {"benchmark", 1, 0, O_BENCH}, {"bench-diff", 1, 0, O_BENCHDIFF}, {"print-interval", 1, 0, O_PRINT},
        {"share-target", 1, 0, O_SHARET}, {"log", 1, 0, O_LOG}, {"debug", 0, 0, O_DEBUG}, {"help", 0, 0, 'h'}, {0, 0, 0, 0}};
    int c;
    while ((c = getopt_long(argc, argv, "o:u:w:p:d:h", opts, NULL)) != -1) {
        switch (c) {
        case 'o':
            if (npools < MAX_POOLS && stratum_parse_url(optarg, &pools[npools]) == 0) npools++;
            else { fprintf(stderr, "bad pool URL: %s\n", optarg); return 1; }
            break;
        case 'u': user = optarg; break;
        case 'w': worker = optarg; break;
        case 'p': pass = optarg; break;
        case 'd': devices = optarg; break;
        case O_KERNEL: kernel = optarg; break;
        case O_VARIANT: g_variant = optarg; break;
        case O_FUSED: g_fused_mode = !strcmp(optarg, "on") ? 1 : !strcmp(optarg, "off") ? -1 : 0; break;
        case O_ROWS: rows = atoi(optarg); break;
        case O_LIST: list = 1; break;
        case O_SELFTEST: selftest = 1; break;
        case O_CPU: use_cpu = 1; break;
        case O_TILE: tile = atoi(optarg); break;
        case O_K: k = atoi(optarg); break;
        case O_COLS: cols = atoi(optarg); break;
        case O_CERT: cert = optarg; break;
        case O_SALTH: salted_h = atoi(optarg); break;
        case O_APIPORT: api_port = atoi(optarg); break;
        case O_APIHOST: api_host = optarg; break;
        case O_BENCH: bench = atoi(optarg); break;
        case O_BENCHDIFF: bench_diff = atof(optarg); break;
        case O_PRINT: print_iv = atoi(optarg); break;
        case O_LOG: logfile = optarg; break;
        case O_SHARET: prl_verify_raw_target = !strcmp(optarg, "raw"); break;
        case O_DEBUG: g_log_level = LOG_DEBUG; break;
        default: usage(); return c == 'h' ? 0 : 1;
        }
    }
    if (logfile) log_open_file(logfile);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    gpu_info gpus[MAX_DEV];
    char err[256] = "";
    int ngpu = use_cpu && !devices ? 0 : gpu_list(gpus, MAX_DEV, err, sizeof err);
    if (list) {
        if (ngpu < 0) { printf("%s\n", err); return 1; }
        for (int i = 0; i < ngpu; i++)
            printf("%d: %s [%s] compute %d.%d, %d SMs, %llu MB\n", i, gpus[i].name, gpus[i].busid, gpus[i].cc_major,
                   gpus[i].cc_minor, gpus[i].sms, gpus[i].mem >> 20);
        return 0;
    }
    if (!use_cpu && ngpu <= 0) { LOGE("no NVIDIA GPUs available: %s", ngpu < 0 ? err : "none found"); return 2; }
    if (!selftest && !bench && (!npools || !user)) { usage(); fprintf(stderr, "\n-o and -u are required\n"); return 1; }

    LOGI("ejjjminer %s — pearlhash, no dev fee, no telemetry", PRL_VERSION);
    if (miner_init(k, cols, tile)) return 1;
    g_miner.cert = !strcmp(cert, "2") ? 2 : !strcmp(cert, "3") ? 3 : 0;
    g_miner.salted_height = salted_h;

    int nd = 0, nsel = 0;
    gpu_dev *gd[MAX_DEV];
    int ngd = 0;
    cpu_dev *cd = NULL;
    if (ngpu > 0) {
        bool sel[MAX_DEV] = {0};
        if (devices) {
            char tmp[256];
            snprintf(tmp, sizeof tmp, "%s", devices);
            for (char *t = strtok(tmp, ", "); t; t = strtok(NULL, ", ")) {
                int v = atoi(t);
                if (v >= 0 && v < ngpu) sel[v] = true;
                else LOGW("ignoring device index %s", t);
            }
        } else
            for (int i = 0; i < ngpu; i++) sel[i] = true;
        for (int i = 0; i < ngpu; i++) {
            if (!sel[i]) continue;
            nsel++;
            gpu_dev *d = gpu_prepare(nd, &gpus[i], kernel, rows);
            if (d) { gd[ngd++] = d; nd++; }
        }
    }
    if (use_cpu) {
        cd = cpu_prepare(nd, 1, rows);
        if (cd) nd++;
    }
    g_miner.ndev = nd;
    if (selftest) {
        LOGI("self-test: %d of %d selected devices passed", ngd, nsel);
        return ngd > 0 ? 0 : 3;
    }
    if (!nd) { LOGE("no usable devices"); return 3; }
    if (miner_start()) return 1;
    for (int i = 0; i < ngd; i++) gpu_launch(gd[i]);
    if (cd) cpu_launch(cd);
    if (api_port > 0) api_start(api_host, api_port);

    double t_end = 0;
    if (bench) {
        g_benchmark = 1;
        uint8_t header[PRL_HEADER_LEN];
        for (int i = 0; i < PRL_HEADER_LEN; i++) header[i] = (uint8_t)(rand() & 0xff);
        u256 target;
        u256_from_double(&target, 26959946667150639794667015087019630673637144422540572481103610249216.0 / bench_diff);
        snprintf(g_miner.pool_url, sizeof g_miner.pool_url, "benchmark");
        miner_on_notify("benchmark", header, &target, 1000000, 0);
        t_end = now_sec() + bench;
        LOGI("benchmark: %d s at share difficulty %g", bench, bench_diff);
    } else {
        stratum_configure(pools, npools, user, worker, pass);
        LOGI("pool: %s  login: %s  cert: %s", stratum_pool(), user, miner_cert_name());
        stratum_start();
    }

    double last_print = now_sec(), last_rate = now_sec();
    while (!g_stop) {
        usleep(250000);
        double t = now_sec();
        if (t - last_rate >= 1.0) {
            for (int i = 0; i < nd; i++) {
                dev_stats *d = &g_miner.dev[i];
                uint64_t w = atomic_load(&d->work);
                double dt = t - (d->last_t ? d->last_t : g_miner.start_t);
                double inst = (double)(w - d->last_work) / (dt > 0 ? dt : 1);
                d->rate = d->last_t ? 0.85 * d->rate + 0.15 * inst : inst;
                d->last_work = w;
                d->last_t = t;
            }
            last_rate = t;
        }
        if (t - last_print >= print_iv) {
            double tot = 0;
            char b[32], line[1024];
            int n = 0;
            for (int i = 0; i < nd; i++) {
                tot += g_miner.dev[i].rate;
                n += snprintf(line + n, sizeof line - n, " | %d: %sH/s", i, fmt_rate(g_miner.dev[i].rate, b, sizeof b));
            }
            LOGI("total %sH/s  A/R/S/I %u/%u/%u/%u%s", fmt_rate(tot, b, sizeof b), atomic_load(&g_miner.shares_accepted),
                 atomic_load(&g_miner.shares_rejected), atomic_load(&g_miner.shares_stale),
                 atomic_load(&g_miner.shares_invalid), line);
            last_print = t;
        }
        if (t_end && t >= t_end) break;
    }
    LOGI("shutting down");
    if (bench) {
        double tot = 0;
        char b[32];
        for (int i = 0; i < nd; i++) tot += (double)atomic_load(&g_miner.dev[i].work);
        LOGI("benchmark result: %sH/s average, %u shares verified", fmt_rate(tot / bench, b, sizeof b),
             atomic_load(&g_miner.shares_accepted));
    }
    g_miner.quit = 1;
    fflush(stdout);
    _exit(atomic_load(&g_miner.shares_invalid) && bench ? 4 : 0);
}
