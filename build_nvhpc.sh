#!/bin/bash
#
# Build script for ASTR using NVIDIA NVHPC (nvfortran) compiler
# Excludes miniapps folder
# Usage: ./build_nvhpc.sh [-j N] [clean]
#

set -e

# ============================================================================
# Parse arguments
# ============================================================================
JOBS=$(nproc)
CLEAN_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -j) JOBS="$2"; shift 2 ;;
        -j*) JOBS="${1#-j}"; shift ;;
        clean) CLEAN_ONLY=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Environment Setup
# ============================================================================
NVHPC_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1
HDF5_HOME=/opt/hdf5-1.14.6

export PATH=${NVHPC_HOME}/compilers/bin:${NVHPC_HOME}/comm_libs/hpcx/bin:${HDF5_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NVHPC_HOME}/compilers/lib:${NVHPC_HOME}/comm_libs/hpcx/lib:${HDF5_HOME}/lib:${LD_LIBRARY_PATH}

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SRCDIR="${PROJECT_ROOT}/src"
UDFDIR="${PROJECT_ROOT}/user_define_module"
OBJDIR="${PROJECT_ROOT}/build_nvhpc_obj"
BINDIR="${PROJECT_ROOT}/bin"

# ============================================================================
# Compiler Configuration
# ============================================================================
FC=h5pfc
FCFLAGS="-Mpreprocess -Mextend -traceback -g -O3 -DHDF5 -Mnostack_arrays"
LDFLAGS="${FCFLAGS}"
LIBS="-lhdf5_hl_fortran -lhdf5_hl -lz -lm"
TARGET="${BINDIR}/astr"

# ============================================================================
# Clean
# ============================================================================
echo "==> Cleaning previous build..."
rm -rf "${OBJDIR}" "${BINDIR}"
mkdir -p "${OBJDIR}" "${BINDIR}"

[[ "$CLEAN_ONLY" == true ]] && echo "==> Clean done." && exit 0

# ============================================================================
# Source file list in dependency order
# ============================================================================
SOURCES=(
    strings.F90 fdnn.F90 singleton.F90 commtype.F90 stlaio.F90 constdef.F90
    tecio.F90 vtkio.F90 interp.F90 commvar.F90 utility.F90 mpiio.F90
    thermchem.F90 commarray.F90 fludyna.F90 parallel.F90 hdf5io.F90
    cmdefne.F90 commfunc.F90 commcal.F90 models.F90 statistic.F90
    userdefine.F90 filter.F90 derivative.F90 flux.F90 bc.F90
    sponge_layer.F90 readwrite.F90 geom.F90 ibmethod.F90 gridgeneration.F90
    riemann.F90 comsolver.F90 solver.F90 udf_pp.F90 pp.F90 initialisation.F90
    mainloop.F90 test.F90 astr.F90
)

TOTAL=${#SOURCES[@]}

# ============================================================================
# Resolve source file paths
# ============================================================================
declare -A SRC_PATH
for src in "${SOURCES[@]}"; do
    if [ -f "${SRCDIR}/${src}" ]; then
        SRC_PATH[$src]="${SRCDIR}/${src}"
    else
        SRC_PATH[$src]="${UDFDIR}/${src}"
    fi
done

# ============================================================================
# Extract: module name -> source file
# ============================================================================
echo "==> Resolving dependencies..."

# Helper: extract module name defined in a Fortran source
get_module() {
    grep -iE '^[[:space:]]*module[[:space:]]+[a-z_]' "$1" 2>/dev/null \
        | tr -d '\r' \
        | sed 's/^[[:space:]]*[Mm][Oo][Dd][Uu][Ll][Ee][[:space:]]*//' \
        | awk '{print $1}' | head -1 || true
}

# Helper: extract USE module names
get_uses() {
    grep -iE '^[[:space:]]*use[[:space:]]+[a-z_]' "$1" 2>/dev/null \
        | tr -d '\r' \
        | sed 's/^[[:space:]]*[Uu][Ss][Ee][[:space:]]*//' \
        | sed 's/[[:space:],].*//' | sort -u || true
}

# Skip list: system/Fortran-intrinsic modules
is_system_module() {
    case "$1" in
        mpi|mpi_f08|mpi_f08_types|hdf5|h5lt|\
        iso_fortran_env|iso_c_binding|\
        ieee_arithmetic|ieee_exceptions|ieee_features|\
        cudafor|openacc|cufft|cublas|cusolver|nvshmem) return 0 ;;
        *) return 1 ;;
    esac
}

# Build module -> source mapping
declare -A MOD_FILE
for src in "${SOURCES[@]}"; do
    mod=$(get_module "${SRC_PATH[$src]}")
    [ -n "$mod" ] && MOD_FILE[$mod]="$src"
done
# Handle mismatches
MOD_FILE[WriteVTK]="vtkio.F90"
MOD_FILE[udf_postprocess]="udf_pp.F90"

# ============================================================================
# Compute per-file .o dependencies
# ============================================================================
declare -A FILE_DEPS

for src in "${SOURCES[@]}"; do
    deps=""
    for mod in $(get_uses "${SRC_PATH[$src]}"); do
        is_system_module "$mod" && continue
        dep_src="${MOD_FILE[$mod]}"
        if [ -n "$dep_src" ] && [ "$dep_src" != "$src" ]; then
            deps="$deps ${OBJDIR}/${dep_src%.F90}.o"
        fi
    done
    FILE_DEPS[$src]="$deps"
done

# ============================================================================
# Generate Makefile
# ============================================================================
MAKEFILE="${OBJDIR}/Makefile.gen"

{
    echo "# Auto-generated Makefile — do not edit"
    echo "# Generated by build_nvhpc.sh"
    echo ""
    echo "FC       = ${FC}"
    echo "FCFLAGS  = ${FCFLAGS}"
    echo "LDFLAGS  = ${LDFLAGS}"
    echo "LIBS     = ${LIBS}"
    echo "OBJDIR   = ${OBJDIR}"
    echo ""
    echo ".DEFAULT_GOAL := all"
    echo ".PHONY: all clean"
    echo ""

    OBJS=""
    for src in "${SOURCES[@]}"; do
        OBJS="${OBJS} ${OBJDIR}/${src%.F90}.o"
    done
    echo "OBJECTS = ${OBJS}"
    echo ""

    echo "TARGET = ${TARGET}"
    echo ""
    echo "all: \$(TARGET)"
    echo ""
    echo "\$(TARGET): \$(OBJECTS)"
    printf "\t@echo '==> Linking...'\n"
    printf "\t\$(FC) \$(LDFLAGS) -o \$@ \$(OBJECTS) \$(LIBS)\n"
    printf "\t@echo '==> Build succeeded!'\n"
    printf "\t@echo '    Executable: \$@'\n"
    echo ""

    # Compile rules with real dependencies
    idx=0
    for src in "${SOURCES[@]}"; do
        idx=$((idx + 1))
        obj="${OBJDIR}/${src%.F90}.o"
        dep_objs="${FILE_DEPS[$src]}"

        echo "# ${src} [${idx}/${TOTAL}]"
        echo "${obj}: ${SRC_PATH[$src]}${dep_objs}"
        printf "\t@echo '  [%s/%s] %s'\n" "${idx}" "${TOTAL}" "${src}"
        printf "\t\$(FC) \$(FCFLAGS) -module \$(OBJDIR) -c -o \$@ \$<\n"
        echo ""
    done

    echo "clean:"
    printf "\trm -rf \$(OBJDIR)/*.o \$(OBJDIR)/*.mod \$(OBJDIR)/*.gen \$(TARGET)\n"

} > "${MAKEFILE}"

# ============================================================================
# Build
# ============================================================================
echo "==> Compiling with ${JOBS} parallel jobs..."
echo "    Compiler: $(which ${FC})"
echo "    Flags:    ${FCFLAGS}"
echo ""

make -f "${MAKEFILE}" -j "${JOBS}"

echo ""
echo "run 'mpirun -np 8 ./bin/astr' to execute the program."
