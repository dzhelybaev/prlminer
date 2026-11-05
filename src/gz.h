#pragma once
#include <stddef.h>
#include <stdint.h>
// gzip (RFC 1952) of `in`; caller frees. NULL on out-of-memory.
uint8_t *prl_gzip(const uint8_t *in, size_t n, size_t *outlen);
