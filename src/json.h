// Tiny JSON DOM parser (enough for stratum messages).
#pragma once
#include <stdbool.h>
#include <stddef.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;
typedef struct jval {
    jtype t;
    bool b;
    double num;
    char *str;           // J_STR
    struct jval *kids;   // J_ARR / J_OBJ children
    char **keys;         // J_OBJ keys
    int n;
} jval;

jval *json_parse(const char *s);   // NULL on error
void json_free(jval *v);
const jval *json_get(const jval *obj, const char *key);
const char *json_str(const jval *v);          // NULL if not a string
bool json_num(const jval *v, double *out);    // numbers or numeric strings
// append a JSON-escaped string (with quotes) to buf
size_t json_escape(char *out, size_t cap, const char *s);
