#include <hdf5.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>

static const char *field_names[] = {"density", "velocity_x", "velocity_y", "velocity_z", "pressure", "temperature",
    "vibrational_temperature", "mass_fraction_N2", "mass_fraction_O2", "mass_fraction_N", "mass_fraction_O", "mass_fraction_NO"};
static const char *derived_names[] = {"velocity_gradient_xx", "velocity_gradient_yx", "velocity_gradient_zx",
    "velocity_gradient_xy", "velocity_gradient_yy", "velocity_gradient_zy", "velocity_gradient_xz",
    "velocity_gradient_yz", "velocity_gradient_zz", "Q_rs", "velocity_divergence", "vorticity_x", "vorticity_y", "vorticity_z"};
static const char *derived_definitions[] = {"du_x/dx", "du_y/dx", "du_z/dx", "du_x/dy", "du_y/dy", "du_z/dy",
    "du_x/dz", "du_y/dz", "du_z/dz", "-0.5*tr(A*A); A(i,j)=du_i/dx_j; full strain",
    "tr(A); A(i,j)=du_i/dx_j", "du_z/dy-du_y/dz", "du_x/dz-du_z/dx", "du_y/dx-du_x/dy"};

static int text_matches(hid_t object, const char *name, const char *expected) {
    hid_t attribute = -1, type = -1, space = -1;
    char value[128];
    size_t length = strlen(expected);
    int ok = length < sizeof(value) && H5Aexists(object, name) > 0;
    if (ok) attribute = H5Aopen(object, name, H5P_DEFAULT);
    if (attribute >= 0) { type = H5Aget_type(attribute); space = H5Aget_space(attribute); }
    ok = ok && type >= 0 && space >= 0;
    if (ok) ok = H5Tget_class(type) == H5T_STRING && H5Tis_variable_str(type) == 0 &&
        H5Tget_size(type) == length && H5Sget_simple_extent_type(space) == H5S_SCALAR &&
        H5Aread(attribute, type, value) >= 0 && !memcmp(value, expected, length);
    if (space >= 0 && H5Sclose(space) < 0) ok = 0;
    if (type >= 0 && H5Tclose(type) < 0) ok = 0;
    if (attribute >= 0 && H5Aclose(attribute) < 0) ok = 0;
    return ok;
}

static hid_t checked_dataset(hid_t group, const char *name, hid_t expected_type, int rank, const hsize_t *dims) {
    hid_t set = -1, type = -1, space = -1;
    H5L_info_t info;
    hsize_t actual[4];
    int ok = H5Lexists(group, name, H5P_DEFAULT) > 0 && H5Lget_info(group, name, &info, H5P_DEFAULT) >= 0 &&
        info.type == H5L_TYPE_HARD;
    if (ok) set = H5Dopen2(group, name, H5P_DEFAULT);
    if (set >= 0) { type = H5Dget_type(set); space = H5Dget_space(set); }
    ok = ok && type >= 0 && space >= 0;
    if (ok) ok = H5Tequal(type, expected_type) > 0 && H5Sget_simple_extent_ndims(space) == rank &&
        H5Sget_simple_extent_dims(space, actual, NULL) >= 0;
    for (int i = 0; ok && i < rank; ++i) ok = actual[i] == dims[i];
    if (space >= 0 && H5Sclose(space) < 0) ok = 0;
    if (type >= 0 && H5Tclose(type) < 0) ok = 0;
    if (!ok && set >= 0) { H5Dclose(set); set = -1; }
    return set;
}

static int read_integers(hid_t group, const char *name, int64_t *values, hsize_t count) {
    hid_t set = checked_dataset(group, name, H5T_STD_I64LE, 1, &count);
    if (set < 0) return 0;
    int ok = H5Dread(set, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) >= 0;
    return H5Dclose(set) >= 0 && ok;
}

static int check_float(hid_t group, const char *name, int rank, const hsize_t *dims, int derivative, const char *units) {
    hid_t set = checked_dataset(group, name, H5T_IEEE_F64LE, rank, dims);
    if (set < 0) return 0;
    int ok = 1;
    if (derivative) {
        const char *quantity = !strcmp(units, "si") ? (derivative == 10 ? "s^-2" : "s^-1") : "dimensionless";
        ok = text_matches(set, "quantity_units", quantity) && text_matches(set, "coordinate_space", "physical") &&
            text_matches(set, "time_dimension_exponent", derivative == 10 ? "-2" : "-1") &&
            text_matches(set, "definition", derived_definitions[derivative-1]);
    }
    return H5Dclose(set) >= 0 && ok;
}

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

/* Read only small metadata. 0: compatible; 2: valid different layout; 1: invalid. */
int astr_output_frame_layout(const char *path, const char *geometry_path, int64_t step, double time,
    int components, const char *units, const int *selected, int nselected, int64_t expected_bytes,
    const int *global_shape, const int *axes, const int *indices, int nplanes,
    int current_components, const char *current_units, const int *current_selected, int current_nselected) {
    hid_t file = -1, geometry = -1, source = -1, coordinates = -1;
    int valid = (components == 6 || components == 12) && (!strcmp(units, "si") || !strcmp(units, "dimensionless")) &&
        (components != 12 || !strcmp(units, "si")) && nselected >= 0 && nselected <= 14;
    int compatible = components == current_components && !strcmp(units, current_units) && nselected == current_nselected;
    int64_t bytes = 0, identity[10], meta[11], geom[11];
    H5G_info_t info, group_info;
    int volume, count;
    char layout[64] = "", label[32], wanted[32];
    for (int i = 0; valid && i < nselected; ++i) {
        valid = selected[i] >= 1 && selected[i] <= 14 && (i == 0 || selected[i] > selected[i-1]);
        if (i < current_nselected) compatible &= selected[i] == current_selected[i];
    }
    if (!valid || astr_checkpoint_hdf5_self_contained(path) || astr_checkpoint_hdf5_self_contained(geometry_path)) return 1;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    geometry = H5Fopen(geometry_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0 || geometry < 0) goto done;
    for (int i = 0; i < 2; ++i) {
        hid_t object = i ? geometry : file;
        if (!text_matches(object, "units", units) || !text_matches(object, "center", "Node") ||
            !text_matches(object, "phase", "completed_step")) goto done;
    }
    if (nselected) {
        size_t used = 0;
        for (int i = 0; i < 14; ++i) used += (size_t)snprintf(layout + used, sizeof(layout) - used,
            "%s%d", i ? " " : "", i < nselected ? selected[i] : 0);
        if (!text_matches(file, "derived_layout", layout)) goto done;
    } else if (H5Aexists(file, "derived_layout") != 0) goto done;
    if (H5Gget_info(file, &info) < 0 || info.nlinks > 769) goto done;
    volume = H5Lexists(file, "metadata", H5P_DEFAULT) > 0;
    count = volume ? 1 : (int)info.nlinks - 1;
    if (count < 1 || count > 768 || (!volume && !read_integers(file, "frame_identity", identity, 10))) goto done;
    compatible &= (volume == (nplanes == 1 && axes[0] == 0)) && count == nplanes;
    for (hsize_t member = 0; member < (volume ? 1 : info.nlinks); ++member) {
        if (volume) label[0] = 0;
        else {
            ssize_t length = H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, member, label, sizeof(label), H5P_DEFAULT);
            if (length < 0 || length >= (ssize_t)sizeof(label)) goto done;
            if (!strcmp(label, "frame_identity")) continue;
        }
        source = volume ? file : H5Gopen2(file, label, H5P_DEFAULT);
        coordinates = volume ? geometry : H5Gopen2(geometry, label, H5P_DEFAULT);
        if (source < 0 || coordinates < 0 || !read_integers(source, "metadata", meta, 11) ||
            !read_integers(coordinates, "metadata", geom, 11)) goto done;
        double clock[3]; memcpy(clock, meta + 2, sizeof(clock));
        if (meta[0] != 1 || meta[1] != step || clock[0] != time || !isfinite(clock[0]) ||
            !isfinite(clock[1]) || !isfinite(clock[2]) || clock[0] < 0 || clock[1] < 0 || clock[2] <= 0 ||
            (step > 0 && clock[1] <= 0) || meta[10] != components || meta[5] < 0 || meta[5] > 3) goto done;
        for (int d = 7; d < 10; ++d) {
            if (meta[d] <= 1) goto done;
            compatible &= meta[d] == global_shape[d-7];
        }
        for (int d = 0; d < 11; ++d) if ((d == 0 || d >= 5) && meta[d] != geom[d]) goto done;
        if (meta[6] < 0 || meta[6] > INT_MAX) goto done;
        int axis = (int)meta[5], index = (int)meta[6];
        if (volume) { if (axis || index) goto done; }
        else {
            if (axis < 1 || index < 0 || meta[6] >= meta[6+axis]) goto done;
            snprintf(wanted, sizeof(wanted), "%c%012lld", "ijk"[axis-1], (long long)meta[6]);
            if (strcmp(wanted, label)) goto done;
            for (int d = 0; d < 5; ++d) if (identity[d] != meta[d]) goto done;
            for (int d = 5; d < 9; ++d) if (identity[d] != meta[d+2]) goto done;
            if (identity[9] != !strcmp(units, "si")) goto done;
        }
        /* Slice input order is presentation, not a different HDF5 layout. */
        int found = 0;
        for (int p = 0; p < nplanes; ++p) found |= axes[p] == axis && indices[p] == index;
        compatible &= found;
        if (H5Gget_info(source, &group_info) < 0 || group_info.nlinks != (hsize_t)(components + nselected + 2) ||
            H5Gget_info(coordinates, &group_info) < 0 || group_info.nlinks != 2) goto done;
        hsize_t dims[4]; int rank = 0; int64_t nodes = 1;
        for (int d = 2; d >= 0; --d) if (axis == 0 || d != axis-1) {
            dims[rank++] = (hsize_t)meta[7+d];
            if (nodes > INT64_MAX / meta[7+d]) goto done;
            nodes *= meta[7+d];
        }
        if (nodes > INT64_MAX / (8 * (components + nselected)) ||
            bytes > INT64_MAX - nodes * 8 * (components + nselected)) goto done;
        bytes += nodes * 8 * (components + nselected);
        for (int i = 0; i < components; ++i) if (!check_float(source, field_names[i], rank, dims, 0, units)) goto done;
        for (int i = 0; i < nselected; ++i)
            if (!check_float(source, derived_names[selected[i]-1], rank, dims, selected[i], units)) goto done;
        dims[rank] = 3;
        if (!check_float(source, "velocity", rank+1, dims, 0, units) ||
            !check_float(coordinates, "coordinates", rank+1, dims, 0, units)) goto done;
        if (!volume) {
            if (H5Gclose(source) < 0) { source = -1; goto done; } source = -1;
            if (H5Gclose(coordinates) < 0) { coordinates = -1; goto done; } coordinates = -1;
        } else { source = -1; coordinates = -1; }
    }
    if (bytes != expected_bytes) goto done;
    valid = 2;
done:
    if (source >= 0 && source != file && H5Gclose(source) < 0) valid = 0;
    if (coordinates >= 0 && coordinates != geometry && H5Gclose(coordinates) < 0) valid = 0;
    if (file >= 0 && H5Fclose(file) < 0) valid = 0;
    if (geometry >= 0 && H5Fclose(geometry) < 0) valid = 0;
    return valid == 2 ? (compatible ? 0 : 2) : 1;
}
