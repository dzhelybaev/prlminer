#pragma once
#include <stddef.h>
int api_start(const char *host, int port);
int api_render(char *out, size_t cap);
