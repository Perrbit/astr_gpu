#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

/* Only the explicitly named test product fails; the solver has no fault hook. */
int link(const char *old_path, const char *new_path) {
    int (*next)(const char *, const char *) = (int (*)(const char *, const char *))dlsym(RTLD_NEXT, "link");
    if (strstr(new_path, "/temporal_products/faulted/") && strstr(new_path, ".complete.csv")) {
        fprintf(stderr, "ASTR_TEST_TEMPORAL_LINK_FAILURE %s\n", new_path);
        errno=EIO;
        return -1;
    }
    return next(old_path,new_path);
}
