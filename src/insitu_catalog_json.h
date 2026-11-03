#ifndef ASTR_INSITU_CATALOG_JSON_H
#define ASTR_INSITU_CATALOG_JSON_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

void *astr_catalog_json_read(const char *, size_t, char *, size_t);
void astr_catalog_json_free(void *);
void *astr_catalog_json_root(void *);
void *astr_catalog_json_get(const void *, const char *);
void *astr_catalog_json_at(const void *, size_t);
void *astr_catalog_json_key(const void *, size_t);
int astr_catalog_json_type(const void *);
size_t astr_catalog_json_size(const void *);
const char *astr_catalog_json_string(const void *, size_t *);
int astr_catalog_json_integer(const void *, int64_t *);
int astr_catalog_json_real(const void *, double *);
int astr_catalog_json_boolean(const void *);
int astr_catalog_display_coordinates(const char *, size_t, int *, double *, char *, size_t);
int astr_catalog_wall_reference(const char *, size_t, int *, double *, char *, size_t);
int astr_catalog_wall_units(const char *, size_t, char *, size_t, char *, size_t);
int astr_catalog_fluctuation(const char *, size_t, int *, char *, size_t);
int astr_catalog_wall_glyph(const char *, size_t, int[2], double *, char *, size_t);
int astr_catalog_wall_trace_seeds(const char *, size_t, int64_t *, size_t, int64_t *, char *, size_t);
int astr_catalog_thickness(const char *, size_t, int *, int64_t *, double *, char *, size_t);
int astr_catalog_file_signature(const char *, int64_t *, char *, size_t);
void *astr_catalog_file_read(const char *, size_t *, int64_t *, char *, size_t);
void astr_catalog_release(void *);
void *astr_catalog_json_quote(const char *, size_t, size_t *);
int astr_catalog_publish_text(const char *, const char *, size_t, char *, size_t);
#ifdef __cplusplus
}
#endif
#endif
