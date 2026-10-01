#include <hdf5.h>

static herr_t inspect_link(hid_t root, const char *name, const H5L_info_t *info, void *context) {
    hid_t object = -1, properties = -1;
    int valid = info->type == H5L_TYPE_HARD;
    (void)context;
    if (!valid) return 1;
    object = H5Oopen(root, name, H5P_DEFAULT);
    if (object < 0) return 1;
    if (H5Iget_type(object) == H5I_DATASET) {
        properties = H5Dget_create_plist(object);
        valid = properties >= 0;
        if (valid) {
            H5D_layout_t layout = H5Pget_layout(properties);
            valid = layout != H5D_LAYOUT_ERROR && H5Pget_external_count(properties) == 0;
#if H5_VERSION_GE(1, 10, 0)
            valid = valid && layout != H5D_VIRTUAL;
#endif
        }
    }
    if (properties >= 0 && H5Pclose(properties) < 0) valid = 0;
    if (H5Oclose(object) < 0) valid = 0;
    return valid ? 0 : 1;
}

int astr_checkpoint_hdf5_self_contained(const char *path) {
    hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return 1;
    herr_t status = H5Lvisit(file, H5_INDEX_NAME, H5_ITER_NATIVE, inspect_link, NULL);
    int valid = status == 0;
    if (H5Fclose(file) < 0) valid = 0;
    return valid ? 0 : 1;
}
