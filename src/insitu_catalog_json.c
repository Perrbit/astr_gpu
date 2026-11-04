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

int astr_catalog_display_coordinates(const char *text, size_t length, int flags[2],
                                     double values[7], char *message, size_t capacity) {
    flags[0] = 1; flags[1] = 0;
    for (int i = 0; i < 7; ++i) values[i] = i < 3 ? 0. : 1.;
    yyjson_doc *doc = astr_catalog_json_read(text, length, message, capacity);
    if (!doc) return 0;
    const yyjson_val *display = yyjson_obj_get(yyjson_doc_get_root(doc), "display");
    const yyjson_val *coordinates = yyjson_obj_get(display, "coordinates");
    const char *error = NULL;
    if (coordinates) {
        if (!yyjson_is_obj(coordinates)) error = "display.coordinates must be an object";
        yyjson_obj_iter iter = yyjson_obj_iter_with(coordinates);
        yyjson_val *key;
        while (!error && (key = yyjson_obj_iter_next(&iter))) {
            const char *name = yyjson_get_str(key);
            if (strcmp(name,"visible") && strcmp(name,"origin") && strcmp(name,"reference_length") &&
                strcmp(name,"display_scale")) error = "unknown display.coordinates parameter";
        }
        const yyjson_val *visible = yyjson_obj_get(coordinates, "visible");
        if (!error && visible) {
            if (!yyjson_is_bool(visible)) error = "coordinate visibility must be a boolean";
            else flags[0] = yyjson_get_bool(visible);
        }
        const yyjson_val *reference = yyjson_obj_get(coordinates, "reference_length");
        if (!error && reference) {
            if (!astr_catalog_json_real(reference, values+3) || values[3]<=0.)
                error = "coordinate reference_length must be positive and finite";
            else flags[1] = 1;
        }
        const yyjson_val *origin = yyjson_obj_get(coordinates, "origin");
        if (!error && origin && !reference) error = "coordinate origin requires reference_length";
        const char *names[2] = {"origin", "display_scale"};
        for (int field=0; field<2 && !error; ++field) {
            const yyjson_val *array = yyjson_obj_get(coordinates,names[field]);
            if (!array) continue;
            if (!yyjson_is_arr(array) || yyjson_arr_size(array)!=3) {
                error = "coordinate origin and display_scale require three numbers"; break;
            }
            for (int d=0; d<3; ++d) {
                double *value = values+(field?4:0)+d;
                if (!astr_catalog_json_real(yyjson_arr_get(array,d),value) || (field && *value<=0.)) {
                    error = "coordinate vectors must be finite; display_scale must be positive"; break;
                }
            }
        }
    }
    yyjson_doc_free(doc);
    if (error) { snprintf(message,capacity,"%s",error); return 0; }
    if (capacity) message[0]='\0';
    return 1;
}

int astr_catalog_wall_reference(const char *text, size_t length, int flags[2],
                                double *value, char *message, size_t capacity) {
    flags[0]=0; flags[1]=0; *value=0.;
    yyjson_doc *doc=astr_catalog_json_read(text,length,message,capacity);
    if (!doc) return 0;
    const yyjson_val *root=yyjson_doc_get_root(doc);
    const yyjson_val *reference=yyjson_obj_get(root,"wall_reference");
    const char *error=NULL;
    if (reference) {
        const char *kind=yyjson_get_str(yyjson_obj_get(root,"kind"));
        if (!kind || strcmp(kind,"wall_profile")) error="wall_reference requires a wall_profile";
        if (!yyjson_is_obj(reference)) error="wall_reference must be an object";
        if (!error) {
            yyjson_obj_iter iter=yyjson_obj_iter_with(reference);
            yyjson_val *key;
            while ((key=yyjson_obj_iter_next(&iter))) {
                const char *name=yyjson_get_str(key);
                if (strcmp(name,"recovery_temperature") && strcmp(name,"recovery_factor")) {
                    error="unknown wall_reference parameter"; break;
                }
            }
        }
        const yyjson_val *temperature=yyjson_obj_get(reference,"recovery_temperature");
        const yyjson_val *factor=yyjson_obj_get(reference,"recovery_factor");
        if (!error && temperature && factor) error="recovery_temperature and recovery_factor are mutually exclusive";
        if (!error && temperature) {
            if (!astr_catalog_json_real(temperature,value) || *value<=0.)
                error="recovery_temperature must be positive and finite in native temperature units";
            else flags[1]=1;
        }
        if (!error && factor) {
            if (!astr_catalog_json_real(factor,value) || *value<0.)
                error="recovery_factor must be nonnegative and finite";
            else flags[1]=2;
        }
        if (!error) flags[0]=1;
    }
    yyjson_doc_free(doc);
    if (error) { snprintf(message,capacity,"%s",error); return 0; }
    if (capacity) message[0]='\0';
    return 1;
}

int astr_catalog_thickness(const char *text, size_t length, int flags[2],
                           int64_t band[2], double *upper, char *message, size_t capacity) {
    flags[0]=flags[1]=0; band[0]=band[1]=0; *upper=0.;
    yyjson_doc *doc=astr_catalog_json_read(text,length,message,capacity);
    if (!doc) return 0;
    const yyjson_val *root=yyjson_doc_get_root(doc);
    const yyjson_val *options=yyjson_obj_get(root,"thickness");
    const char *error=NULL;
    if (options) {
        const char *kind=yyjson_get_str(yyjson_obj_get(root,"kind"));
        const char *source=yyjson_get_str(yyjson_obj_get(root,"source"));
        if (!kind || strcmp(kind,"station_profile") || !source || strcmp(source,"favre"))
            error="thickness requires a Favre station_profile";
        else if (!yyjson_is_obj(options)) error="thickness must be an object";
        if (!error) {
            yyjson_obj_iter iter=yyjson_obj_iter_with(options);
            yyjson_val *key;
            while ((key=yyjson_obj_iter_next(&iter))) {
                const char *name=yyjson_get_str(key);
                if (strcmp(name,"edge_band") && strcmp(name,"explicit_upper")) {
                    error="unknown thickness parameter"; break;
                }
            }
        }
        const yyjson_val *range=yyjson_obj_get(options,"edge_band");
        if (!error && (!yyjson_is_arr(range) || yyjson_arr_size(range)!=2 ||
            !astr_catalog_json_integer(yyjson_arr_get(range,0),band) ||
            !astr_catalog_json_integer(yyjson_arr_get(range,1),band+1) || band[0]<1 || band[1]<band[0]))
            error="edge_band requires ordered global j indices strictly above the wall at j=0";
        const yyjson_val *limit=yyjson_obj_get(options,"explicit_upper");
        if (!error && limit) {
            if (!astr_catalog_json_real(limit,upper)) error="explicit_upper must be a finite native y coordinate";
            else flags[1]=1;
        }
        if (!error) flags[0]=1;
    }
    yyjson_doc_free(doc);
    if (error) { snprintf(message,capacity,"%s",error); return 0; }
    if (capacity) message[0]='\0';
    return 1;
}

int astr_catalog_fluctuation(const char *text, size_t length, int *flag,
                             char *message, size_t capacity) {
    *flag=0;
    yyjson_doc *doc=astr_catalog_json_read(text,length,message,capacity);
    if (!doc) return 0;
    const yyjson_val *value=yyjson_obj_get(yyjson_doc_get_root(doc),"fluctuation");
    const char *error=NULL;
    if (value) {
        if (!yyjson_is_bool(value)) error="fluctuation must be a boolean";
        else *flag=yyjson_get_bool(value);
    }
    yyjson_doc_free(doc);
    if (error) { snprintf(message,capacity,"%s",error); return 0; }
    if (capacity) message[0]='\0';
    return 1;
}

int astr_catalog_temporal(const char *text, size_t length, int64_t parameters[3], int *archive,
                          char *message, size_t capacity) {
    parameters[0]=1;parameters[1]=parameters[2]=0;*archive=0;
    yyjson_doc *doc=astr_catalog_json_read(text,length,message,capacity);
    if (!doc) return 0;
    const yyjson_val *root=yyjson_doc_get_root(doc),*options=yyjson_obj_get(root,"temporal");
    const char *kind=yyjson_get_str(yyjson_obj_get(root,"kind")),*error=NULL;
    const int line=kind && !strcmp(kind,"time_space"),spectrum=kind && !strcmp(kind,"probe_spectrum");
    if (line || spectrum) {
        if (!yyjson_is_obj(options)) error="temporal products require a temporal object";
        else {
            yyjson_obj_iter iter=yyjson_obj_iter_with(options);yyjson_val *key;
            while ((key=yyjson_obj_iter_next(&iter))) {
                const char *name=yyjson_get_str(key);
                if (strcmp(name,"sample_steps") && strcmp(name,"segment_length") &&
                    strcmp(name,"block_samples") && strcmp(name,"archive_samples")) {
                    error="unknown temporal parameter";break;
                }
            }
        }
        const char *names[]={"sample_steps","segment_length","block_samples"};
        for (int i=0;i<3 && !error;++i) {
            const yyjson_val *value=yyjson_obj_get(options,names[i]);
            if (value && (!astr_catalog_json_integer(value,parameters+i) || parameters[i]<1 ||
                (i>0 && parameters[i]>INT_MAX))) error="temporal counts must be positive representable integers";
        }
        const yyjson_val *value=yyjson_obj_get(options,"archive_samples");
        if (!error && value) {
            if (!yyjson_is_bool(value)) error="archive_samples must be a boolean";
            else *archive=yyjson_get_bool(value);
        }
        if (!error && line && (parameters[2]<1 || parameters[1]!=0 || value))
            error="time_space requires block_samples; all line history is archived without segment_length/archive_samples";
        if (!error && spectrum && (parameters[1]<2 || parameters[1]%2))
            error="probe_spectrum requires an even segment_length of at least two";
        if (!error && spectrum && ((*archive && parameters[2]<1) || (!*archive && parameters[2]!=0)))
            error="probe block_samples is required exactly when archive_samples is true";
    } else if (options) error="temporal options require time_space or probe_spectrum";
    yyjson_doc_free(doc);
    if (error) {snprintf(message,capacity,"%s",error);return 0;}
    if (capacity) message[0]='\0';
    return 1;
}

int astr_catalog_wall_glyph(const char *text, size_t length, int stride[2], double *native_length,
                             char *message, size_t capacity) {
    stride[0]=stride[1]=0; *native_length=0.;
    yyjson_doc *doc=astr_catalog_json_read(text,length,message,capacity);
    if (!doc) return 0;
    const yyjson_val *root=yyjson_doc_get_root(doc);
    const char *field=yyjson_get_str(yyjson_obj_get(root,"field"));
    const char *error=NULL;
    if (field && !strcmp(field,"wall_shear_vector")) {
        const yyjson_val *options=yyjson_obj_get(root,"extraction");
        if (!yyjson_is_obj(options)) error="wall_shear_vector requires extraction with stride and length";
        else {
            yyjson_obj_iter iter=yyjson_obj_iter_with(options);yyjson_val *key;
            while ((key=yyjson_obj_iter_next(&iter))) {
                const char *name=yyjson_get_str(key);
                if (strcmp(name,"stride") && strcmp(name,"length")) {error="unknown wall glyph parameter";break;}
            }
        }
        const yyjson_val *steps=yyjson_obj_get(options,"stride");
        int64_t values[2]={0,0};
        if (!error && (!yyjson_is_arr(steps) || yyjson_arr_size(steps)!=2 ||
            !astr_catalog_json_integer(yyjson_arr_get(steps,0),values) ||
            !astr_catalog_json_integer(yyjson_arr_get(steps,1),values+1) ||
            values[0]<1 || values[0]>INT_MAX || values[1]<1 || values[1]>INT_MAX))
            error="wall glyph stride requires two positive representable i/k index intervals";
        if (!error && (!astr_catalog_json_real(yyjson_obj_get(options,"length"),native_length) || *native_length<=0.))
            error="wall glyph length must be positive and finite in native coordinate units";
        if (!error) {stride[0]=(int)values[0];stride[1]=(int)values[1];}
    }
    yyjson_doc_free(doc);
    if (error) {snprintf(message,capacity,"%s",error);return 0;}
    if (capacity) message[0]='\0';
    return 1;
}

int astr_catalog_wall_trace_seeds(const char *text, size_t length, int64_t *indices,
                                size_t index_capacity, int64_t *count, char *message, size_t capacity) {
    *count=0;
    yyjson_doc *doc=astr_catalog_json_read(text,length,message,capacity);
    if (!doc) return 0;
    const yyjson_val *root=yyjson_doc_get_root(doc);
    const char *field=yyjson_get_str(yyjson_obj_get(root,"field"));
    const char *error=NULL;
    if (field && !strcmp(field,"wall_shear_lines")) {
        const yyjson_val *seeds=yyjson_obj_get(yyjson_obj_get(root,"extraction"),"seeds");
        const size_t n=yyjson_arr_size(seeds);
        if (!yyjson_is_arr(seeds) || !n || n>(size_t)INT_MAX/4)
            error="wall shear lines require a nonempty representable list of global [i,k] seeds";
        if (!error && indices && index_capacity<2*n) error="wall seed buffer is smaller than its declared list";
        for (size_t i=0;!error && i<n;++i) {
            const yyjson_val *seed=yyjson_arr_get(seeds,i);
            int64_t values[2];
            if (!yyjson_is_arr(seed) || yyjson_arr_size(seed)!=2 ||
                !astr_catalog_json_integer(yyjson_arr_get(seed,0),values) ||
                !astr_catalog_json_integer(yyjson_arr_get(seed,1),values+1) ||
                values[0]<0 || values[1]<0 || values[0]>INT_MAX || values[1]>INT_MAX)
                error="wall seeds must be nonnegative representable global [i,k] integer pairs";
            if (!error && indices) {indices[2*i]=values[0];indices[2*i+1]=values[1];}
        }
        if (!error) *count=(int64_t)n;
    }
    yyjson_doc_free(doc);
    if (error) {snprintf(message,capacity,"%s",error);return 0;}
    if (capacity) message[0]='\0';
    return 1;
}

int astr_catalog_wall_units(const char *text, size_t length, char *target,
                           size_t target_capacity, char *message, size_t capacity) {
    if (target_capacity) target[0]='\0';
    yyjson_doc *doc=astr_catalog_json_read(text,length,message,capacity);
    if (!doc) return 0;
    const yyjson_val *root=yyjson_doc_get_root(doc);
    const yyjson_val *reference=yyjson_obj_get(root,"wall_units");
    const char *error=NULL;
    if (reference) {
        const char *kind=yyjson_get_str(yyjson_obj_get(root,"kind"));
        const char *name=yyjson_get_str(reference);
        if (!kind || strcmp(kind,"station_profile")) error="wall_units requires a station_profile";
        else if (!name || !name[0] || strlen(name)!=yyjson_get_len(reference))
            error="wall_units must name a nonempty wall_profile product ID";
        else {
            for (size_t i=0; i<strlen(name); ++i) {
                const unsigned char c=(unsigned char)name[i];
                if (c<32 || c==127) { error="wall_units product ID must not contain control characters"; break; }
            }
            if (!error && strlen(name)>=target_capacity) error="wall_units destination is too small";
            if (!error) memcpy(target,name,strlen(name)+1);
        }
    }
    yyjson_doc_free(doc);
    if (error) { snprintf(message,capacity,"%s",error); return 0; }
    if (capacity) message[0]='\0';
    return 1;
}
