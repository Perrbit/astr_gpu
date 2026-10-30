#define _POSIX_C_SOURCE 200809L
#include "insitu_catalog_json.h"
#include "vendor/yyjson/yyjson.h"
#include <limits.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void file_identity(const struct stat *s, int64_t identity[8]) {
    identity[0] = (int64_t)s->st_dev;
    identity[1] = (int64_t)s->st_ino;
    identity[2] = (int64_t)s->st_size;
    identity[3] = (int64_t)s->st_mtim.tv_sec;
    identity[4] = (int64_t)s->st_mtim.tv_nsec;
    identity[5] = (int64_t)s->st_ctim.tv_sec;
    identity[6] = (int64_t)s->st_ctim.tv_nsec;
    identity[7] = (int64_t)s->st_mode;
}

int astr_catalog_file_signature(const char *path, int64_t *identity, char *message, size_t capacity) {
    struct stat s;
    int code = 0;
    memset(identity, 0, 8 * sizeof(*identity));
    if (stat(path, &s)) code = errno;
    else if (!S_ISREG(s.st_mode)) code = EINVAL;
    if (code) {
        snprintf(message, capacity, "%s", strerror(code));
        return code;
    }
    file_identity(&s, identity);
    if (capacity) message[0] = '\0';
    return 0;
}

void *astr_catalog_file_read(const char *path, size_t *length, int64_t *identity,
                            char *message, size_t message_capacity) {
    struct stat s;
    FILE *stream = fopen(path, "rb");
    char *data = NULL;
    size_t used = 0, capacity = 4096;
    int code = stream ? 0 : errno;
    *length = 0;
    memset(identity, 0, 8 * sizeof(*identity));
    if (stream && fstat(fileno(stream), &s)) code = errno;
    if (!code && !S_ISREG(s.st_mode)) code = EINVAL;
    if (!code) {
        file_identity(&s, identity);
        data = malloc(capacity + 1);
        if (!data) code = ENOMEM;
    }
    while (!code) {
        used += fread(data + used, 1, capacity - used, stream);
        if (ferror(stream)) { code = errno ? errno : EIO; break; }
        if (feof(stream)) break;
        if (capacity > (SIZE_MAX - 1) / 2) { code = EOVERFLOW; break; }
        char *larger = realloc(data, 2 * capacity + 1);
        if (!larger) { code = ENOMEM; break; }
        data = larger;
        capacity *= 2;
    }
    if (stream && fclose(stream) && !code) code = errno ? errno : EIO;
    if (code) {
        free(data);
        snprintf(message, message_capacity, "%s", strerror(code));
        return NULL;
    }
    data[used] = '\0';
    *length = used;
    if (message_capacity) message[0] = '\0';
    return data;
}

void astr_catalog_release(void *value) { free(value); }

void *astr_catalog_json_quote(const char *text, size_t length, size_t *encoded_length) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) return NULL;
    yyjson_mut_val *value = yyjson_mut_strn(doc, length ? text : "", length);
    if (!value) { yyjson_mut_doc_free(doc); return NULL; }
    yyjson_mut_doc_set_root(doc, value);
    char *encoded = yyjson_mut_write(doc, 0, encoded_length);
    yyjson_mut_doc_free(doc);
    return encoded;
}

int astr_catalog_publish_text(const char *path, const char *text, size_t length,
                              char *message, size_t message_capacity) {
    size_t n = strlen(path) + 16;
    char *temporary = malloc(n);
    int code = 0, fd = -1;
    FILE *stream = NULL;
    if (!temporary) code = ENOMEM;
    if (!code) {
        snprintf(temporary, n, "%s.tmp.XXXXXX", path);
        fd = mkstemp(temporary);
        if (fd < 0) code = errno;
    }
    if (!code) {
        stream = fdopen(fd, "wb");
        if (!stream) { code = errno; close(fd); }
    }
    if (stream) {
        if (fwrite(text, 1, length, stream) != length) code = errno ? errno : EIO;
        if (fclose(stream) && !code) code = errno ? errno : EIO;
    }
    if (!code && rename(temporary, path)) code = errno;
    if (code) {
        if (fd >= 0) unlink(temporary);
        snprintf(message, message_capacity, "%s", strerror(code));
    } else if (message_capacity) message[0] = '\0';
    free(temporary);
    return code;
}

static int duplicate_keys(const yyjson_val *value) {
    if (yyjson_is_obj(value)) {
        yyjson_obj_iter outer = yyjson_obj_iter_with(value);
        yyjson_val *key;
        while ((key = yyjson_obj_iter_next(&outer))) {
            yyjson_obj_iter rest = outer;
            yyjson_val *other;
            while ((other = yyjson_obj_iter_next(&rest))) {
                if (yyjson_get_len(key) == yyjson_get_len(other) &&
                    !memcmp(yyjson_get_str(key), yyjson_get_str(other), yyjson_get_len(key))) return 1;
            }
            if (duplicate_keys(yyjson_obj_iter_get_val(key))) return 1;
        }
    } else if (yyjson_is_arr(value)) {
        yyjson_arr_iter iter = yyjson_arr_iter_with(value);
        yyjson_val *child;
        while ((child = yyjson_arr_iter_next(&iter))) if (duplicate_keys(child)) return 1;
    }
    return 0;
}

void *astr_catalog_json_read(const char *text, size_t length, char *message, size_t capacity) {
    yyjson_read_err error;
    yyjson_doc *doc = yyjson_read_opts((char *)text, length, 0, NULL, &error);
    if (!doc) {
        snprintf(message, capacity, "JSON byte %zu: %s", error.pos, error.msg);
        return NULL;
    }
    if (duplicate_keys(yyjson_doc_get_root(doc))) {
        snprintf(message, capacity, "duplicate JSON object key");
        yyjson_doc_free(doc);
        return NULL;
    }
    if (capacity) message[0] = '\0';
    return doc;
}

void astr_catalog_json_free(void *doc) { yyjson_doc_free(doc); }
void *astr_catalog_json_root(void *doc) { return yyjson_doc_get_root(doc); }
void *astr_catalog_json_get(const void *obj, const char *key) { return yyjson_obj_get(obj, key); }
void *astr_catalog_json_at(const void *arr, size_t index) { return yyjson_arr_get(arr, index); }
void *astr_catalog_json_key(const void *obj, size_t index) {
    yyjson_obj_iter iter = yyjson_obj_iter_with(obj);
    yyjson_val *key = NULL;
    for (size_t i = 0; i <= index; ++i) if (!(key = yyjson_obj_iter_next(&iter))) break;
    return key;
}

int astr_catalog_json_type(const void *value) {
    const yyjson_val *v = value;
    if (yyjson_is_obj(v)) return 1;
    if (yyjson_is_arr(v)) return 2;
    if (yyjson_is_str(v)) return 3;
    if (yyjson_is_int(v)) return 4;
    if (yyjson_is_real(v)) return 5;
    if (yyjson_is_bool(v)) return 6;
    if (yyjson_is_null(v)) return 7;
    return 0;
}

size_t astr_catalog_json_size(const void *value) {
    return yyjson_is_obj(value) ? yyjson_obj_size(value) : yyjson_arr_size(value);
}

const char *astr_catalog_json_string(const void *value, size_t *length) {
    *length = yyjson_get_len(value);
    return yyjson_get_str(value);
}

int astr_catalog_json_integer(const void *value, int64_t *result) {
    if (!yyjson_is_int(value)) return 0;
    if (yyjson_is_uint(value) && yyjson_get_uint(value) > INT64_MAX) return 0;
    *result = yyjson_get_sint(value);
    return 1;
}

int astr_catalog_json_real(const void *value, double *result) {
    if (!yyjson_is_num(value)) return 0;
    *result = yyjson_get_num(value);
    return isfinite(*result);
}

int astr_catalog_json_boolean(const void *value) {
    if (!yyjson_is_bool(value)) return -1;
    return yyjson_get_bool(value) ? 1 : 0;
}
