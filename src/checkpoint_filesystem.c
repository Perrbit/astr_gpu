#define _GNU_SOURCE
#ifdef __linux__
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int plain_at(int fd, const char *name) {
    struct stat st;
    return !fstatat(fd, name, &st, AT_SYMLINK_NOFOLLOW) && S_ISREG(st.st_mode) && st.st_nlink == 1;
}

/* Validate every path component before normalizing, including link/.. aliases. */
static int plain_directory_path(const char *path) {
    char copy[PATH_MAX], *save = NULL;
    int fd, next;
    if (!path[0] || strlen(path) >= sizeof(copy) || strchr(path, ':')) return -1;
    for (const unsigned char *p = (const unsigned char *)path; *p; ++p)
        if (*p < 32 || *p == 127) return -1;
    strcpy(copy, path);
    fd = open(path[0] == '/' ? "/" : ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return -1;
    for (char *p = strtok_r(copy, "/", &save); p; p = strtok_r(NULL, "/", &save)) {
        next = openat(fd, p, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (close(fd) || next < 0) { if (next >= 0) close(next); return -1; }
        fd = next;
    }
    return fd;
}

int astr_output_archive_parent(const char *current, const char *parent, char *relative, int capacity) {
    char a[PATH_MAX], b[PATH_MAX], result[2 * PATH_MAX];
    int fd = -1, source = -1, status = 1;
    size_t common = 0, used = 0;
    if (capacity < 1) return 1;
    relative[0] = 0;
    fd = plain_directory_path(current); source = plain_directory_path(parent);
    if (fd < 0 || source < 0 || !plain_at(source, "SEGMENT") || !plain_at(source, "input.txt") ||
        !realpath(current, a) || !realpath(parent, b) || !strcmp(a, b)) goto done;
    for (size_t i = 0; a[i] && b[i] && a[i] == b[i]; ++i) if (a[i] == '/') common = i + 1;
    /* One ../ for each remaining directory, followed by the parent's suffix. */
    for (size_t i = common; a[i]; ++i) {
        if (i == common || a[i] == '/') { memcpy(result + used, "../", 3); used += 3; }
    }
    if (used + strlen(b + common) + 1 > sizeof(result) ||
        used + strlen(b + common) + 1 > (size_t)capacity) goto done;
    strcpy(result + used, b + common);
    strcpy(relative, result); status = 0;
done:
    if (source >= 0 && close(source)) status = 1;
    if (fd >= 0 && close(fd)) status = 1;
    return status;
}

int astr_checkpoint_reuse_run(const char *root, const char *restore, int *reuse) {
    int fd = -1, resources = -1, checkpoints = -1, source = -1, parent = -1, result = 1;
    int has_resources, has_checkpoints;
    struct stat resource_stat, checkpoint_stat, parent_stat;
    *reuse = 0;
    fd = open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return 1;
    has_resources = !fstatat(fd, "resources", &resource_stat, AT_SYMLINK_NOFOLLOW);
    if (!has_resources && errno != ENOENT) goto done;
    has_checkpoints = !fstatat(fd, "checkpoints", &checkpoint_stat, AT_SYMLINK_NOFOLLOW);
    if (!has_checkpoints && errno != ENOENT) goto done;
    if (!has_resources && !has_checkpoints) { result = 0; goto done; }
    if (!has_resources || !has_checkpoints || !restore[0]) goto done;
    resources = openat(fd, "resources", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    checkpoints = openat(fd, "checkpoints", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    source = open(restore, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (resources < 0 || checkpoints < 0 || source < 0) goto done;
    parent = openat(source, "..", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (parent < 0 || fstat(checkpoints, &checkpoint_stat) || fstat(parent, &parent_stat)) goto done;
    if (checkpoint_stat.st_dev != parent_stat.st_dev || checkpoint_stat.st_ino != parent_stat.st_ino) goto done;
    *reuse = 1; result = 0;
done:
    if (parent >= 0 && close(parent)) result = 1;
    if (source >= 0 && close(source)) result = 1;
    if (checkpoints >= 0 && close(checkpoints)) result = 1;
    if (resources >= 0 && close(resources)) result = 1;
    if (close(fd)) result = 1;
    return result;
}

int astr_output_archive_segment(const char *root, int reuse, int *generation, int *fresh) {
    int fd = -1, child = -1, result = 1, high = -1;
    DIR *dir = NULL;
    struct dirent *entry;
    struct stat st;
    *generation = -1; *fresh = 0;
    if (!mkdir(root, 0700)) { *generation = 0; *fresh = 1; return 0; }
    if (errno != EEXIST || !reuse) return 1;
    fd = open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return 1;
    child = openat(fd, "resources", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (child < 0 || !plain_at(child, "data.h5")) goto done;
    if (close(child)) { child = -1; goto done; }
    child = -1;
    dir = fdopendir(fd);
    if (!dir) goto done;
    fd = -1;
    errno = 0;
    while ((entry = readdir(dir))) {
        const char *name = entry->d_name;
        if (!strcmp(name, ".") || !strcmp(name, "..") || !strcmp(name, "resources")) continue;
        if (strlen(name) != 15 || strncmp(name, "segment", 7)) goto done;
        int number = 0;
        for (int i = 7; i < 15; ++i) {
            if (name[i] < '0' || name[i] > '9') goto done;
            number = number * 10 + name[i] - '0';
        }
        if (fstatat(dirfd(dir), name, &st, AT_SYMLINK_NOFOLLOW) || !S_ISDIR(st.st_mode)) goto done;
        child = openat(dirfd(dir), name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (child < 0 || !plain_at(child, "SEGMENT") || !plain_at(child, "input.txt") ||
            !plain_at(child, "series.frames")) goto done;
        if (close(child)) { child = -1; goto done; }
        child = -1;
        if (number > high) high = number;
        errno = 0;
    }
    if (errno || high < 0 || high == 99999999) goto done;
    *generation = high + 1; result = 0;
done:
    if (child >= 0 && close(child)) result = 1;
    if (dir && closedir(dir)) result = 1;
    if (fd >= 0 && close(fd)) result = 1;
    return result;
}

int astr_checkpoint_publication_root(const char *root) {
    int fd = open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC), result = 1;
    struct stat st;
    if (fd < 0) return 1;
    if (!fstatat(fd, ".LATEST.tmp", &st, AT_SYMLINK_NOFOLLOW) || errno != ENOENT) goto done;
    if (!fstatat(fd, "LATEST", &st, AT_SYMLINK_NOFOLLOW)) {
        if (!S_ISREG(st.st_mode) || st.st_nlink != 1) goto done;
    } else if (errno != ENOENT) goto done;
    result = 0;
done:
    if (close(fd)) result = 1;
    return result;
}

int astr_checkpoint_replace_plain_file(const char *root, const char *temporary, const char *target) {
    int fd = -1, result = 1;
    struct stat st;
    if (!temporary[0] || !target[0] || strchr(temporary, '/') || strchr(target, '/') ||
        !strcmp(temporary, target) || !strcmp(temporary, ".") || !strcmp(temporary, "..") ||
        !strcmp(target, ".") || !strcmp(target, "..")) return 1;
    fd = open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return 1;
    if (fstatat(fd, temporary, &st, AT_SYMLINK_NOFOLLOW) || !S_ISREG(st.st_mode) || st.st_nlink != 1) goto done;
    if (!fstatat(fd, target, &st, AT_SYMLINK_NOFOLLOW)) {
        if (!S_ISREG(st.st_mode) || st.st_nlink != 1) goto done;
    } else if (errno != ENOENT) goto done;
    result = renameat(fd, temporary, fd, target) != 0;
done:
    if (close(fd)) result = 1;
    return result;
}

int astr_checkpoint_inflow_count(const char *path, int *count) {
    int result = 1, high = -1, fd;
    unsigned char *seen = calloc(100000, 1);
    DIR *dir = NULL;
    struct dirent *entry;
    struct stat st;
    *count = 0;
    if (!seen) return 1;
    fd = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) goto done;
    dir = fdopendir(fd);
    if (!dir) { close(fd); goto done; }
    errno = 0;
    while ((entry = readdir(dir))) {
        const char *name = entry->d_name;
        if (strncmp(name, "islice", 6)) continue;
        if (strlen(name) != 14 || strcmp(name + 11, ".h5")) goto done;
        int number = 0;
        for (int i = 6; i < 11; ++i) {
            if (name[i] < '0' || name[i] > '9') goto done;
            number = 10 * number + name[i] - '0';
        }
        if (seen[number] || fstatat(dirfd(dir), name, &st, AT_SYMLINK_NOFOLLOW) ||
            !S_ISREG(st.st_mode) || st.st_nlink != 1) goto done;
        seen[number] = 1;
        if (number > high) high = number;
        errno = 0;
    }
    if (errno || high < 3) goto done;
    for (int i = 0; i <= high; ++i) if (!seen[i]) goto done;
    *count = high + 1;
    result = 0;
done:
    if (dir && closedir(dir)) result = 1;
    free(seen);
    return result;
}

int astr_checkpoint_plain_resource(const char *batch, const char *name) {
    int b = -1, root = -1, resources = -1, file = -1, result = 1;
    struct stat st;
    if (!name[0] || name[0] == '.' || strchr(name, '/')) return 1;
    b = open(batch, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (b < 0) goto done;
    if (!strcmp(name, "SEGMENT")) {
        resources = openat(b, "..", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (resources < 0) goto done;
        goto member;
    }
    root = openat(b, "../..", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (root < 0) goto done;
    resources = openat(root, "resources", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (resources < 0) goto done;
member:
    file = openat(resources, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    if (file < 0) goto done;
    if (!fstat(file, &st) && S_ISREG(st.st_mode) && st.st_nlink == 1) result = 0;
done:
    if (file >= 0) close(file);
    if (resources >= 0) close(resources);
    if (root >= 0) close(root);
    if (b >= 0) close(b);
    return result;
}

/* No recursive deletion or path following. The caller has validated CRCs and
 * must serialize publication/retirement within its private output directory. */
int astr_checkpoint_retire(const char *root, const char *name, const char *entries) {
    char names[66][129], latest[130], list[66 * 130];
    int count = 0, rootfd = -1, batchfd = -1, latestfd = -1, result = 1, seen = 0, scanfd;
    ssize_t n;
    DIR *dir = NULL;
    struct dirent *entry;
    struct stat st;
    if (!name[0] || strchr(name, '/') || name[0] == '.' || strlen(entries) >= sizeof(list)) return 1;
    strcpy(list, entries);
    char *save = NULL;
    for (char *p = strtok_r(list, "\n", &save); p; p = strtok_r(NULL, "\n", &save)) {
        if (count == 66 || !p[0] || strlen(p) > 128 || strchr(p, '/') || p[0] == '.') return 1;
        for (int i = 0; i < count; ++i) if (!strcmp(names[i], p)) return 1;
        strcpy(names[count++], p);
    }
    if (count < 3) return 1;
    rootfd = open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (rootfd < 0) goto done;
    latestfd = openat(rootfd, "LATEST", O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (latestfd < 0) goto done;
    n = read(latestfd, latest, sizeof(latest) - 1);
    if (n <= 0 || n >= (ssize_t)sizeof(latest) - 1) goto done;
    latest[n] = 0;
    if (latest[n - 1] != '\n') goto done;
    latest[n - 1] = 0;
    if (!strcmp(latest, name)) goto done;
    batchfd = openat(rootfd, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (batchfd < 0) goto done;
    /* A protection marker excludes the batch from ordinary retention. */
    if (fstatat(batchfd, "PROTECT", &st, AT_SYMLINK_NOFOLLOW) == 0) {
        result = 2;
        goto done;
    }
    if (errno != ENOENT) goto done;
    scanfd = dup(batchfd);
    if (scanfd < 0) goto done;
    dir = fdopendir(scanfd);
    if (!dir) {
        close(scanfd);
        goto done;
    }
    errno = 0;
    while ((entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        int found = 0;
        for (int i = 0; i < count; ++i) if (!strcmp(entry->d_name, names[i])) found = 1;
        if (!found || fstatat(batchfd, entry->d_name, &st, AT_SYMLINK_NOFOLLOW) ||
            !S_ISREG(st.st_mode) || st.st_nlink != 1) goto done;
        ++seen;
        errno = 0;
    }
    if (errno || seen != count) goto done;
    /* Invalidate first, so an interrupted cleanup is never a valid restore. */
    if (unlinkat(batchfd, "COMPLETE", 0)) goto done;
    for (int i = 0; i < count; ++i) {
        if (!strcmp(names[i], "COMPLETE")) continue;
        if (unlinkat(batchfd, names[i], 0)) goto done;
    }
    if (unlinkat(rootfd, name, AT_REMOVEDIR)) goto done;
    result = 0;
done:
    if (dir) closedir(dir);
    if (latestfd >= 0) close(latestfd);
    if (batchfd >= 0) close(batchfd);
    if (rootfd >= 0) close(rootfd);
    return result;
}
#else
int astr_output_archive_parent(const char *current, const char *parent, char *relative, int capacity) {
    (void)current; (void)parent;
    if (capacity > 0) relative[0] = 0;
    return 1;
}
int astr_checkpoint_reuse_run(const char *root, const char *restore, int *reuse) {
    (void)root; (void)restore; *reuse = 0;
    return 1;
}
int astr_output_archive_segment(const char *root, int reuse, int *generation, int *fresh) {
    (void)root; (void)reuse; *generation = -1; *fresh = 0;
    return 1;
}
int astr_checkpoint_publication_root(const char *root) {
    (void)root;
    return 1;
}
int astr_checkpoint_replace_plain_file(const char *root, const char *temporary, const char *target) {
    (void)root; (void)temporary; (void)target;
    return 1;
}
int astr_checkpoint_inflow_count(const char *path, int *count) {
    (void)path;
    *count = 0;
    return 1;
}
int astr_checkpoint_plain_resource(const char *batch, const char *name) {
    (void)batch;
    (void)name;
    return 1;
}
int astr_checkpoint_retire(const char *root, const char *name, const char *entries) {
    (void)root;
    (void)name;
    (void)entries;
    return 1;
}
#endif
