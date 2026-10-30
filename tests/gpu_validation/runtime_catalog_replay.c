#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Publish only at the real root watcher entry; no solver hook or flow mutation. */
int stat(const char *path, struct stat *result) {
    int (*next)(const char *, struct stat *) = dlsym(RTLD_NEXT, "stat");
    const char *formal = getenv("ASTR_TEST_CATALOG_FILE");
    const char *directory = getenv("ASTR_TEST_CATALOG_REPLAY");
    const char *rank = getenv("OMPI_COMM_WORLD_RANK");
    static unsigned polls;
    if (formal && directory && rank && !strcmp(rank, "0") && !strcmp(path, formal)) {
        char candidate[PATH_MAX];
        ++polls;
        int n = snprintf(candidate, sizeof(candidate), "%s/catalog%06u.jsonl", directory, polls);
        if (n < 0 || n >= (int)sizeof(candidate)) { errno = ENAMETOOLONG; return -1; }
        if (!rename(candidate, path)) {
            dprintf(STDERR_FILENO, "ASTR_TEST_CATALOG_PUBLISH poll=%u\n", polls);
        } else if (errno != ENOENT) {
            return -1;
        }
    }
    return next(path, result);
}

int linkat(int old_directory, const char *old_path, int new_directory, const char *new_path, int flags) {
    int (*next)(int, const char *, int, const char *, int) = dlsym(RTLD_NEXT, "linkat");
    const char *geometry = getenv("ASTR_TEST_GEOMETRY_FAIL_ID");
    const char *geometry_step = getenv("ASTR_TEST_GEOMETRY_FAIL_STEP");
    const char *kind = getenv("ASTR_TEST_GEOMETRY_FAIL_KIND");
    const char *target_rank = getenv("ASTR_TEST_GEOMETRY_FAIL_RANK");
    const char *rank = getenv("OMPI_COMM_WORLD_RANK");
    if (geometry && geometry_step && kind && target_rank && rank && !strcmp(rank, target_rank)) {
        char product[PATH_MAX], frame[128];
        snprintf(product, sizeof(product), "/geometry_products/%s/", geometry);
        if (!strcmp(kind, "piece"))
            snprintf(frame, sizeof(frame), "frame.step%012ld.rank%08ld.vtp", strtol(geometry_step, NULL, 10), strtol(rank, NULL, 10));
        else snprintf(frame, sizeof(frame), "frame.step%012ld.pvtp", strtol(geometry_step, NULL, 10));
        size_t n = strlen(new_path), tail = strlen(frame);
        if (strstr(new_path, product) && n >= tail && !strcmp(new_path + n - tail, frame)) { errno = EIO; return -1; }
    }
    const char *id = getenv("ASTR_TEST_IMAGE_FAIL_ID");
    const char *step = getenv("ASTR_TEST_IMAGE_FAIL_STEP");
    if (id && step) {
        char product[PATH_MAX], frame[64];
        snprintf(product, sizeof(product), "/render_products/%s/", id);
        snprintf(frame, sizeof(frame), "frame.step%012ld.jpeg", strtol(step, NULL, 10));
        size_t n = strlen(new_path), tail = strlen(frame);
        if (strstr(new_path, product) && n >= tail && !strcmp(new_path + n - tail, frame)) {
            errno = EIO;
            return -1;
        }
    }
    return next(old_directory, old_path, new_directory, new_path, flags);
}

int rename(const char *old_path, const char *new_path) {
    int (*next)(const char *, const char *) = dlsym(RTLD_NEXT, "rename");
    const char *id = getenv("ASTR_TEST_WALL_FAIL_ID");
    const char *step = getenv("ASTR_TEST_WALL_FAIL_STEP");
    if (id && step) {
        char product[PATH_MAX], frame[64];
        snprintf(product, sizeof(product), "/wall_products/%s/", id);
        snprintf(frame, sizeof(frame), "frame.step%012ld.csv", strtol(step, NULL, 10));
        size_t n = strlen(new_path), tail = strlen(frame);
        if (strstr(new_path, product) && n >= tail && !strcmp(new_path + n - tail, frame)) {
            errno = EIO;
            return -1;
        }
    }
    return next(old_path, new_path);
}
