#!/bin/bash
#
# Build script for ASTR using NVIDIA NVHPC (nvfortran) compiler
# Excludes miniapps folder
#

set -e

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
# Use h5pfc which wraps NVHPC mpifort + HDF5 includes/libs
FC=h5pfc

# NVHPC-specific flags (replacing gfortran flags)
# -Mpreprocess : enable C-style preprocessing (needed for #ifdef HDF5 etc.)
# -Mextend     : allow 132-column source lines (replaces -ffree-line-length-none)
# -traceback   : generate traceback on error (replaces -fbacktrace)
# -g           : include debug symbols
# -O3          : optimization level
# -DHDF5       : enable HDF5 code paths
# -Kieee       : strict IEEE floating-point conformance
# -Mnobounds   : disable bounds checking for performance (remove for debug builds)
FCFLAGS="-Mpreprocess -Mextend -traceback -O0 -DHDF5 -Kieee"

# Module output directory (replaces -J)
MODULE_FLAG="-module"

# Libraries
# Note: h5pfc only adds -lhdf5_fortran -lhdf5 by default.
# h5lt (HDF5 Light) functions need explicit addition:
LIBS="-lhdf5_hl_fortran -lhdf5_hl -lz -lm"

# ============================================================================
# Build Target
# ============================================================================
TARGET="${BINDIR}/astr"

# ============================================================================
# Source Files (compilation order respects module dependencies)
# ============================================================================
# Order from Makefile.astr — modules must be compiled before files that USE them
SOURCES=(
    strings.F90
    fdnn.F90
    singleton.F90
    commtype.F90
    stlaio.F90
    constdef.F90
    tecio.F90
    vtkio.F90
    interp.F90
    commvar.F90
    utility.F90
    mpiio.F90
    thermchem.F90
    commarray.F90
    fludyna.F90
    parallel.F90
    hdf5io.F90
    cmdefne.F90
    commfunc.F90
    commcal.F90
    models.F90
    statistic.F90
    userdefine.F90
    filter.F90
    derivative.F90
    flux.F90
    bc.F90
    sponge_layer.F90
    readwrite.F90
    geom.F90
    ibmethod.F90
    gridgeneration.F90
    riemann.F90
    comsolver.F90
    solver.F90
    udf_pp.F90
    pp.F90
    initialisation.F90
    mainloop.F90
    test.F90
    astr.F90
)

# ============================================================================
# Clean previous build
# ============================================================================
echo "==> Cleaning previous build..."
rm -rf "${OBJDIR}" "${BINDIR}"
mkdir -p "${OBJDIR}" "${BINDIR}"

# ============================================================================
# Compile each source file
# ============================================================================
echo "==> Compiling source files with NVHPC..."
echo "    Compiler: $(which ${FC})"
echo "    Flags:    ${FCFLAGS}"
echo ""

OBJECTS=()
for src in "${SOURCES[@]}"; do
    # Determine which directory the file is in
    if [ -f "${SRCDIR}/${src}" ]; then
        srcpath="${SRCDIR}/${src}"
    elif [ -f "${UDFDIR}/${src}" ]; then
        srcpath="${UDFDIR}/${src}"
    else
        echo "ERROR: Cannot find source file: ${src}"
        exit 1
    fi

    objname="${src%.F90}.o"
    objpath="${OBJDIR}/${objname}"

    echo "  [$((${#OBJECTS[@]}+1))/${#SOURCES[@]}] Compiling ${src}..."

    ${FC} ${FCFLAGS} ${MODULE_FLAG} "${OBJDIR}" -c -o "${objpath}" "${srcpath}"

    OBJECTS+=("${objpath}")
done

# ============================================================================
# Link executable
# ============================================================================
echo ""
echo "==> Linking executable..."
${FC} ${FCFLAGS} -o "${TARGET}" "${OBJECTS[@]}" ${LIBS}

# ============================================================================
# Report
# ============================================================================
echo ""
echo "========================================================================"
echo "  Build succeeded!"
echo "  Executable: ${TARGET}"
echo "  Objects:    ${OBJDIR}"
echo "========================================================================"
echo "run 'mpirun -np 8 ./bin/astr' to execute the program."