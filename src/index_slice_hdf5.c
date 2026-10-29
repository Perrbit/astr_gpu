#define _POSIX_C_SOURCE 200809L
#include <hdf5.h>
#include <mpi.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>

int astr_index_slice_directory(const char *root) {
    char *path = strdup(root);
    if (!path) return ENOMEM;
    int code = 0;
    for (char *p = path + 1; ; ++p) {
        if (*p != '/' && *p) continue;
        char saved = *p; *p = '\0';
        if (mkdir(path, 0700) && errno != EEXIST) code = errno;
        *p = saved;
        if (code || !saved) break;
    }
    free(path);
    return code;
}

int astr_index_slice_segment(const char *root, char *segment, size_t capacity) {
    int code = astr_index_slice_directory(root);
    if (code) return code;
    int n = snprintf(segment, capacity, "%s/segmentXXXXXX", root);
    if (n < 0 || (size_t)n >= capacity) return ENAMETOOLONG;
    if (!mkdtemp(segment)) return errno;
    return 0;
}

static int agree(MPI_Comm comm, int local) {
    int global;
    if (MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS)
        MPI_Abort(comm, 71);
    return global;
}

static int consistent_handle(MPI_Comm comm, hid_t handle) {
    int present = handle >= 0, least, most;
    if (MPI_Allreduce(&present, &least, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS ||
        MPI_Allreduce(&present, &most, 1, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS)
        MPI_Abort(comm, 71);
    /* A partially created collective object cannot safely be closed by all ranks. */
    if (least != most) MPI_Abort(comm, 72);
    return least;
}

static int text_attribute(hid_t object, const char *name, const char *value, MPI_Comm comm) {
    hid_t space = -1, type = -1, attribute = -1;
    int ok;
    space = H5Screate(H5S_SCALAR);
    type = H5Tcopy(H5T_C_S1);
    ok = space >= 0 && type >= 0;
    if (ok) ok = H5Tset_size(type, strlen(value) ? strlen(value) : 1) >= 0;
    if (ok) ok = H5Tset_cset(type, H5T_CSET_UTF8) >= 0;
    if (!agree(comm, ok)) goto done;
    attribute = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    ok = consistent_handle(comm, attribute);
    if (ok) ok = agree(comm, H5Awrite(attribute, type, value) >= 0);
done:
    if (attribute >= 0) ok = (H5Aclose(attribute) >= 0) && ok;
    if (type >= 0) ok = (H5Tclose(type) >= 0) && ok;
    if (space >= 0) ok = (H5Sclose(space) >= 0) && ok;
    return agree(comm, ok);
}

static int small_vector(hid_t file, const char *name, hid_t stored, hid_t native,
                        const void *values, hsize_t count, MPI_Comm comm, int rank) {
    hid_t space = -1, set = -1;
    int ok;
    space = H5Screate_simple(1, &count, NULL);
    ok = agree(comm, space >= 0);
    if (!ok) goto done;
    set = H5Dcreate2(file, name, stored, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ok = consistent_handle(comm, set);
    if (!ok) goto done;
    ok = rank != 0 || H5Dwrite(set, native, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) >= 0;
    ok = agree(comm, ok);
done:
    if (set >= 0) ok = (H5Dclose(set) >= 0) && ok;
    if (space >= 0) ok = (H5Sclose(space) >= 0) && ok;
    return agree(comm, ok);
}

/* Arrays contain only this rank's unique slice nodes; no global field gather. */
int astr_index_slice_write(const char *path, const char *geometry, int geometry_only,
        int fortran_comm, int bytes, const int *dimensions, const int *offset,
        const int *extent, int nfields, const char *names, const char *units,
        const int64_t *field_ids, const void *values, const void *coordinates,
        int64_t step, const double *clock, const double *references, const int64_t *ranges, const char *definition,
        const char *provenance) {
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)fortran_comm);
    hid_t access = -1, file = -1, transfer = -1, space = -1, memory = -1, set = -1;
    hid_t native = bytes == 4 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
    hid_t stored = bytes == 4 ? H5T_IEEE_F32LE : H5T_IEEE_F64LE;
    hsize_t shape[3] = {(hsize_t)dimensions[1], (hsize_t)dimensions[0], 3};
    hsize_t start[3] = {(hsize_t)offset[1], (hsize_t)offset[0], 0};
    hsize_t count[3] = {(hsize_t)extent[1], (hsize_t)extent[0], 3};
    hsize_t nodes = (hsize_t)extent[0] * extent[1], memory_count;
    int rank, ok = 0, stage = 1, field, dim, coordinate;
    char name[33], unit[33];
    H5E_auto2_t error_handler;
    void *error_context;
    MPI_Comm_rank(comm, &rank);
    H5Eget_auto2(H5E_DEFAULT, &error_handler, &error_context);
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    access = H5Pcreate(H5P_FILE_ACCESS);
    if (!agree(comm, access >= 0)) goto done;
    if (!agree(comm, H5Pset_fapl_mpio(access, comm, MPI_INFO_NULL) >= 0)) goto done;
    file = H5Fcreate(path, H5F_ACC_EXCL, H5P_DEFAULT, access);
    if (!consistent_handle(comm, file)) goto done;
    stage = 2;
    if (!text_attribute(file, "phase", "completed_step", comm) ||
        !text_attribute(file, "product_definition", definition, comm) ||
        !text_attribute(file, "source_measure", provenance, comm) ||
        !text_attribute(file, "geometry_file", geometry, comm) ||
        !text_attribute(file, "reference_state_layout", "roinf uinf vinf winf pinf tinf ref_den ref_vel ref_tem ref_len ref_tim ref_miu", comm) ||
        !text_attribute(file, "center", "Node", comm)) goto done;
    if (!small_vector(file, "step", H5T_STD_I64LE, H5T_NATIVE_INT64, &step, 1, comm, rank) ||
        !small_vector(file, "clock", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, clock, 3, comm, rank) ||
        !small_vector(file, "reference_state", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, references, 12, comm, rank) ||
        !small_vector(file, "index_ranges", H5T_STD_I64LE, H5T_NATIVE_INT64, ranges, 6, comm, rank) ||
        !small_vector(file, "field_ids", H5T_STD_I64LE, H5T_NATIVE_INT64, field_ids, nfields, comm, rank)) goto done;
    transfer = H5Pcreate(H5P_DATASET_XFER);
    if (!agree(comm, transfer >= 0)) goto done;
    if (!agree(comm, H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) >= 0)) goto done;
    stage = 3;
    for (field = 0; field < (geometry_only ? 1 : nfields); ++field) {
        coordinate = geometry_only;
        dim = coordinate ? 3 : 2;
        if (coordinate) strcpy(name, "coordinates");
        else {
            memcpy(name, names + 32 * field, 32); name[32] = '\0';
            for (int n = 31; n >= 0 && name[n] == ' '; --n) name[n] = '\0';
        }
        space = H5Screate_simple(dim, shape, NULL);
        if (!agree(comm, space >= 0)) goto done;
        set = H5Dcreate2(file, name, stored, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (!consistent_handle(comm, set)) goto done;
        if (!coordinate) {
            memcpy(unit, units + 32 * field, 32); unit[32] = '\0';
            for (int n = 31; n >= 0 && unit[n] == ' '; --n) unit[n] = '\0';
            if (!text_attribute(set, "units", unit, comm)) goto done;
        }
        memory_count = nodes * (coordinate ? 3 : 1);
        if (!memory_count) memory_count = 1;
        memory = H5Screate_simple(1, &memory_count, NULL);
        if (!agree(comm, memory >= 0)) goto done;
        ok = nodes ? H5Sselect_hyperslab(space, H5S_SELECT_SET, start, NULL, count, NULL) >= 0
                   : H5Sselect_none(space) >= 0 && H5Sselect_none(memory) >= 0;
        if (!agree(comm, ok)) goto done;
        const char empty[8] = {0};
        const void *data = nodes ? (coordinate ? coordinates : (const char *)values + (size_t)field * nodes * bytes) : empty;
        ok = agree(comm, H5Dwrite(set, native, memory, space, transfer, data) >= 0);
        int closed = H5Sclose(memory) >= 0; memory = -1;
        closed = (H5Dclose(set) >= 0) && closed; set = -1;
        closed = (H5Sclose(space) >= 0) && closed; space = -1;
        if (!agree(comm, ok && closed)) goto done;
    }
    stage = 0;
done:
    ok = stage == 0;
    if (memory >= 0) ok = (H5Sclose(memory) >= 0) && ok;
    if (set >= 0) ok = (H5Dclose(set) >= 0) && ok;
    if (space >= 0) ok = (H5Sclose(space) >= 0) && ok;
    if (transfer >= 0) ok = (H5Pclose(transfer) >= 0) && ok;
    if (file >= 0) ok = (H5Fclose(file) >= 0) && ok;
    if (access >= 0) ok = (H5Pclose(access) >= 0) && ok;
    ok = agree(comm, ok);
    H5Eset_auto2(H5E_DEFAULT, error_handler, error_context);
    return ok ? 0 : (stage ? stage : 4);
}
