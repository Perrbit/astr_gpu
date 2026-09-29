ASTR Code

Version 2.6

ASTR is a high-order finite-difference flow solver designed for high-fidelity simulation of compressible turbulence. It supports multi-physics extensions including combustion and is optimized for modern high-performance computing systems.

Dependencies
Before building ASTR, ensure the following dependencies are installed:

Fortran 90 compiler (e.g., gfortran, ifort)
MPI (e.g., OpenMPI, MPICH)
HDF5 with Fortran bindings
(Optional) Cantera (for combustion simulations)

Download the Source Code
Clone the official repository:
git clone git@github.com:astr-code/astr.git

Compilation and Installation

Option 1: Using make
A simple build using GNU Make:

cd astr
make
The executable will be located at:
./bin/astr

For the current CUDA Fortran implementation, see the Chinese
[user guide](USER_GUIDE.md) for deployment, input files, boundary conditions,
MPI/GPU configuration, restart, optional features, and supported limits.
The legacy CPU instructions below do not define GPU feature availability.

Option 2: Using CMake (Recommended)
CMake provides a safer and more flexible build environment. To compile and install using CMake:

For a production build without the added validation sources and probe targets,
configure with `-DBUILD_TESTING=OFF`. The `tests/` directory may then be omitted
from the source package. Keep `examples/` and `user_define_module/`: the original
example configuration and production source dependencies remain unchanged.
`BUILD_TESTING=ON` (the default) retains development probes and the added
boundary-validation commands. CUDA-aware MPI detection is available in either
mode. Reconfigure existing build directories explicitly when switching modes.

With a compatible NVHPC, MPI and HDF5 environment, a CUDA production build is:

```bash
cmake -S . -B build_prod -DCMAKE_Fortran_COMPILER=nvfortran \
  -DASTR_WITH_CUDA=ON -DBUILD_TESTING=OFF
cmake --build build_prod --target astr -j 2
```

This enables CUDA support in the binary; `usegpu` remains an input-file choice.
Fixed AIR5 support is a separate opt-in build option,
`-DASTR_WITH_AIR5_CHEMISTRY=ON`, and is not the legacy Cantera `CHEMISTRY` option.

Create a case directory:

mkdir test_case

cmake path_to_the_source

cmake --build .

cmake --install .

ctest -L nondim

The binary will be installed under:

test_case/opt/bin/astr

The default install prefix is `<build-directory>/opt`. Override it with
`-DCMAKE_INSTALL_PREFIX=/absolute/install/path` or
`cmake --install <build-directory> --prefix /absolute/install/path`.
The build-tree executable remains `<build-directory>/bin/astr`.
The legacy Make, Cantera and `script/install.sh` instructions are not the
minimal GPU release workflow; those tools may be absent from a trimmed package.
Use the root CMake build and `examples/GPU_Quickstart/` for that package.

Enabling Combustion Module
ASTR supports detailed chemical kinetics via Cantera. To enable this feature:

Install Cantera (Fortran interface required):

python scons/scripts/scons.py build python_package=none \
    FORTRAN=<your fortran compiler> f90_interface=y \
    prefix=<installation_dir> boost_inc_dir=<boost_include_dir>

python scons/scripts/scons.py test
python scons/scripts/scons.py install

Configure ASTR with Cantera support:
cmake -DCHEMISTRY=TRUE -DCANTERA_DIR=path_to_cantera path_to_the_source

cmake --build .

cmake --install .

ctest -L combustion

Running a Simulation
To execute a simulation:
mpirun -np 8 ./astr run ./datin/input_file

To install astr for a new case, a user can use the intall astr in the script as
path_to_astr/script/install.sh
<replace files in user_define_module as required>
make
<create files in the datin>
<run a simulation>

Mini Apps
ASTR includes lightweight mini-applications for testing and development.

To compile and run mini-apps:

mkdir test_mini_apps
cd test_mini_apps
cmake path_to_the_source/miniapps/
cmake --build .
./astr.min



Directory Structure Overview

astr/
├── src/                # Core solver source code
├── script/             # Utility scripts (e.g. case creation)
├── miniapps/           # Testing and development tools
├── pastr/              # pre and post-processing module for astr
    ├── src/            # src of pastr
├── bin/                # Compiled binaries
└── examples/           # Sample cases

Contact
For questions, bug reports, or contributions, please open an issue on GitHub or contact the development team.

Let me know if you want to include badges (e.g., build status), citation info, or extended documentation links.
