#include "json.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct { const char *p; int depth; } jp;

static void ws(jp *s) { while (*s->p == ' ' || *s->p == '\t' || *s->p == '\n' || *s->p == '\r') s->p++; }

static bool parse_val(jp *s, jval *v);

static char *parse_str_raw(jp *s) {
    if (*s->p != '"') return NULL;
    s->p++;
    size_t cap = 64, n = 0;
    char *o = malloc(cap);
    while (*s->p && *s->p != '"') {
        char c = *s->p++;
        if (c == '\\') {
            char e = *s->p++;
            switch (e) {
            case 'n': c = '\n'; break;
            case 't': c = '\t'; break;
            case 'r': c = '\r'; break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'u': {
                unsigned u = 0;
                for (int i = 0; i < 4 && *s->p; i++) {
                    char h = *s->p++;
                    u = u * 16 + (h >= '0' && h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
                }
                c = u < 128 ? (char)u : '?';
                break;
            }
            case 0: free(o); return NULL;
            default: c = e;
            }
        }
        if (n + 2 > cap) { cap *= 2; o = realloc(o, cap); }
        o[n++] = c;
    }
    if (*s->p != '"') { free(o); return NULL; }
    s->p++;
    o[n] = 0;
    return o;
}

static bool parse_val(jp *s, jval *v) {
    memset(v, 0, sizeof(*v));
    ws(s);
    if (++s->depth > 64) return false;
    bool ok = true;
    char c = *s->p;
    if (c == '{' || c == '[') {
        bool obj = c == '{';
        v->t = obj ? J_OBJ : J_ARR;
        s->p++;
        int cap = 0;
        ws(s);
        if (*s->p == (obj ? '}' : ']')) { s->p++; s->depth--; return true; }
        for (;;) {
            ws(s);
            char *key = NULL;
            if (obj) {
                key = parse_str_raw(s);
                if (!key) { ok = false; break; }
                ws(s);
                if (*s->p != ':') { free(key); ok = false; break; }
                s->p++;
            }
            if (v->n == cap) {
                cap = cap ? cap * 2 : 8;
                v->kids = realloc(v->kids, cap * sizeof(jval));
                if (obj) v->keys = realloc(v->keys, cap * sizeof(char *));
            }
            if (obj) v->keys[v->n] = key;
            if (!parse_val(s, &v->kids[v->n])) { v->n++; ok = false; break; }
            v->n++;
            ws(s);
            if (*s->p == ',') { s->p++; continue; }
            if (*s->p == (obj ? '}' : ']')) { s->p++; break; }
            ok = false;
            break;
        }
    } else if (c == '"') {
        v->t = J_STR;
        v->str = parse_str_raw(s);
        ok = v->str != NULL;
    } else if (!strncmp(s->p, "true", 4)) { v->t = J_BOOL; v->b = true; s->p += 4; }
    else if (!strncmp(s->p, "false", 5)) { v->t = J_BOOL; s->p += 5; }
    else if (!strncmp(s->p, "null", 4)) { v->t = J_NULL; s->p += 4; }
    else {
        char *end;
        v->t = J_NUM;
        v->num = strtod(s->p, &end);
        if (end == s->p) ok = false;
        s->p = end;
    }
    s->depth--;
    return ok;
}

jval *json_parse(const char *str) {
    jp s = {str, 0};
    jval *v = calloc(1, sizeof(jval));
    if (!parse_val(&s, v)) { json_free(v); return NULL; }
    return v;
}

static void free_inner(jval *v) {
    free(v->str);
    for (int i = 0; i < v->n; i++) {
        free_inner(&v->kids[i]);
        if (v->keys) free(v->keys[i]);
    }
    free(v->kids);
    free(v->keys);
}

void json_free(jval *v) {
    if (!v) return;
    free_inner(v);
    free(v);
}

const jval *json_get(const jval *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->n; i++)
        if (!strcmp(o->keys[i], key)) return &o->kids[i];
    return NULL;
}

const char *json_str(const jval *v) { return v && v->t == J_STR ? v->str : NULL; }

bool json_num(const jval *v, double *out) {
    if (!v) return false;
    if (v->t == J_NUM) { *out = v->num; return true; }
    if (v->t == J_STR && v->str[0]) {
        char *e;
        double d = strtod(v->str, &e);
        if (*e == 0) { *out = d; return true; }
    }
    return false;
}

size_t json_escape(char *out, size_t cap, const char *s) {
    size_t n = 0;
#define PUT(c) do { if (n + 1 < cap) out[n] = (c); n++; } while (0)
    PUT('"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { PUT('\\'); PUT(c); }
        else if (c < 0x20) {
            char t[8];
            snprintf(t, sizeof t, "\\u%04x", c);
            for (char *p = t; *p; p++) PUT(*p);
        } else PUT(c);
    }
    PUT('"');
    if (cap) out[n < cap ? n : cap - 1] = 0;
#undef PUT
    return n;
}
