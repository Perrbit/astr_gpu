#define _GNU_SOURCE
#include <hdf5.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

herr_t H5Dwrite(hid_t set, hid_t type, hid_t memory, hid_t space, hid_t transfer, const void *values) {
    static herr_t (*real_write)(hid_t,hid_t,hid_t,hid_t,hid_t,const void *) = NULL;
    if (!real_write) *(void **)(&real_write) = dlsym(RTLD_NEXT, "H5Dwrite");
    if (getenv("ASTR_TEST_SLICE_WRITE_FAILURE")) {
        char filename[4096], name[128];
        hid_t file = H5Iget_file_id(set);
        H5Fget_name(file, filename, sizeof(filename));
        H5Iget_name(set, name, sizeof(name));
        H5Fclose(file);
        if (strstr(filename, "/faulty/") && strstr(filename, "/step000000000004.h5.tmp") &&
            !strcmp(name, "/velocity_x")) return -1;
    }
    return real_write(set, type, memory, space, transfer, values);
}
