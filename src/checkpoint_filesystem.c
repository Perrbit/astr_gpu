#define _GNU_SOURCE
#ifdef __linux__
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int astr_checkpoint_plain_resource(const char *batch, const char *name) {
    int b = -1, root = -1, resources = -1, file = -1, result = 1;
    struct stat st;
    if (!name[0] || name[0] == '.' || strchr(name, '/')) return 1;
    b = open(batch, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (b < 0) goto done;
    root = openat(b, "../..", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (root < 0) goto done;
    resources = openat(root, "resources", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (resources < 0) goto done;
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
