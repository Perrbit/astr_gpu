#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Deterministic, test-owned controller reads; the solver and its reload cadence
 * are unchanged. The initial read remains stale deliberately to test dt_next. */
static const char *controller_path(const char *path, char *replacement, size_t size) {
    const char *source = getenv("ASTR_CONTROLLER_TEST_FILE");
    const char *replay = getenv("ASTR_CONTROLLER_TEST_REPLAY");
    const char *start = getenv("ASTR_CONTROLLER_TEST_START_STEP");
    char actual[PATH_MAX], *end;
    static unsigned reads;
    if (!source || !replay || !start || !realpath(path, actual) || strcmp(source, actual)) return path;
    errno = 0;
    long first = strtol(start, &end, 10);
    if (errno || *end || first < 0 || first > 1000000 || reads > 1000000) { errno = EINVAL; return NULL; }
    long completed = reads ? first + reads : 0;
    if (snprintf(replacement, size, "%s/controller%012ld", replay, completed) >= (int)size) {
        errno = ENAMETOOLONG;
        return NULL;
    }
    reads++;
    dprintf(STDERR_FILENO, "ASTR_CONTROLLER_TEST_READ completed_step=%ld\n", completed);
    return replacement;
}

#define CONTROLLER_OPEN(symbol) \
int symbol(const char *path, int flags, ...) { \
    mode_t mode = 0; \
    if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) { \
        va_list args; va_start(args, flags); mode = va_arg(args, int); va_end(args); \
    } \
    int (*real)(const char *, int, ...) = dlsym(RTLD_NEXT, #symbol); \
    if (!real) { errno = ENOSYS; return -1; } \
    char replacement[PATH_MAX]; \
    /* NVHPC action='read' without status='old' opens an existing file O_RDWR. */ \
    int input = (flags & O_ACCMODE) == O_RDONLY || \
        ((flags & O_ACCMODE) == O_RDWR && !(flags & (O_CREAT | O_TRUNC | O_APPEND))); \
    const char *selected = input ? \
        controller_path(path, replacement, sizeof(replacement)) : path; \
    if (!selected) return -1; \
    return real(selected, flags, mode); \
}

CONTROLLER_OPEN(open)
CONTROLLER_OPEN(open64)

#define CONTROLLER_FOPEN(symbol) \
FILE *symbol(const char *path, const char *mode) { \
    FILE *(*real)(const char *, const char *) = dlsym(RTLD_NEXT, #symbol); \
    if (!real) { errno = ENOSYS; return NULL; } \
    char replacement[PATH_MAX]; \
    /* NVHPC maps the existing action='read', status='unknown' input to r+. */ \
    const char *selected = mode[0] == 'r' ? \
        controller_path(path, replacement, sizeof(replacement)) : path; \
    return selected ? real(selected, mode) : NULL; \
}

CONTROLLER_FOPEN(fopen)
CONTROLLER_FOPEN(fopen64)

static int phase(const char *wanted) {
    const char *value = getenv("ASTR_CHECKPOINT_TEST_FAULT");
    return value && !strcmp(value, wanted);
}

static int directory_matches(int fd, const char *suffix) {
    char descriptor[64], actual[1400], expected[1400];
    const char *root = getenv("ASTR_CHECKPOINT_TEST_ROOT");
    if (!root) return 0;
    snprintf(descriptor, sizeof(descriptor), "/proc/self/fd/%d", fd);
    ssize_t length = readlink(descriptor, actual, sizeof(actual) - 1);
    if (length < 0 || length == (ssize_t)sizeof(actual) - 1) return 0;
    actual[length] = 0;
    if (snprintf(expected, sizeof(expected), "%s%s", root, suffix) >= (int)sizeof(expected)) return 0;
    return !strcmp(actual, expected);
}

static int target_matches(int fd, const char *path, const char *suffix) {
    const char *target = getenv("ASTR_CHECKPOINT_TEST_TARGET");
    const char *root = getenv("ASTR_CHECKPOINT_TEST_ROOT");
    char expected[1400], actual[1400], cwd[1024];
    const char *destination = path;
    if (path[0] != '/' && fd == AT_FDCWD && getcwd(cwd, sizeof(cwd)) &&
        snprintf(actual, sizeof(actual), "%s/%s", cwd, path) < (int)sizeof(actual)) destination = actual;
    return root && target &&
        snprintf(expected, sizeof(expected), "%s/%s%s", root, target, suffix) < (int)sizeof(expected) &&
        !strcmp(destination, expected);
}

int mkdir(const char *path, mode_t mode) {
    if (phase("batch_create") && target_matches(AT_FDCWD, path, ".tmp")) {
        errno = EIO;
        return -1;
    }
    int (*real)(const char *, mode_t) = dlsym(RTLD_NEXT, "mkdir");
    if (!real) { errno = ENOSYS; return -1; }
    return real(path, mode);
}

int renameat2(int oldfd, const char *old, int newfd, const char *new, unsigned flags) {
    if (phase("batch_rename") && target_matches(newfd, new, "")) {
        errno = EIO;
        return -1;
    }
    int (*real)(int, const char *, int, const char *, unsigned) = dlsym(RTLD_NEXT, "renameat2");
    if (!real) { errno = ENOSYS; return -1; }
    int result = real(oldfd, old, newfd, new, flags);
    /* Preserve a published test checkpoint before later retention, without a
     * timing-sensitive external watcher or any change to solver state. */
    if (!result && phase("protect_batch") && target_matches(newfd, new, "")) {
        int batch = openat(newfd, new, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (batch < 0) return -1;
        int marker = openat(batch, "PROTECT", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        int saved = errno;
        if (marker >= 0 && close(marker)) { marker = -1; saved = errno; }
        if (close(batch) && marker >= 0) return -1;
        if (marker < 0) { errno = saved; return -1; }
    }
    return result;
}

int renameat(int oldfd, const char *old, int newfd, const char *new) {
    if (phase("latest_rename") && !strcmp(old, ".LATEST.tmp") && !strcmp(new, "LATEST") &&
        directory_matches(oldfd, "")) {
        const char *target = getenv("ASTR_CHECKPOINT_TEST_TARGET");
        char value[130];
        int fd = openat(oldfd, old, O_RDONLY | O_NOFOLLOW);
        ssize_t count = fd >= 0 ? read(fd, value, sizeof(value) - 1) : -1;
        if (fd >= 0) close(fd);
        if (count > 0 && target) {
            value[count] = 0;
            if (value[count - 1] == '\n') value[count - 1] = 0;
            if (!strcmp(value, target)) { errno = EIO; return -1; }
        }
    }
    int (*real)(int, const char *, int, const char *) = dlsym(RTLD_NEXT, "renameat");
    if (!real) { errno = ENOSYS; return -1; }
    return real(oldfd, old, newfd, new);
}

int unlinkat(int fd, const char *name, int flags) {
    const char *retired = getenv("ASTR_CHECKPOINT_TEST_RETIRE");
    char suffix[130];
    if (!retired) retired = "batch1";
    if (snprintf(suffix, sizeof(suffix), "/%s", retired) < (int)sizeof(suffix) && directory_matches(fd, suffix) &&
        ((phase("retire_marker") && !strcmp(name, "COMPLETE")) ||
         (phase("retire_payload") && !strcmp(name, "state.h5")))) {
        errno = EIO;
        return -1;
    }
    int (*real)(int, const char *, int) = dlsym(RTLD_NEXT, "unlinkat");
    if (!real) { errno = ENOSYS; return -1; }
    return real(fd, name, flags);
}
