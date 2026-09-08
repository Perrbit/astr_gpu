Overview: This dataset contains the result of the Mach 2 2D laminar shock-wave/boundary-layer interaction simulation, performed in the automatic source code generation framework OpenSBLI. 

Contents: opensbli.h5
The file is in the Hierarchical Data Format (HDF5), and contains 7 datasets from the simulation. The datasets are for the 2D conservative variables of the compressible Navier-Stokes equations, spatial coordinates and a dataset of skin-friction computed along the no-slip wall. For spatial coordinates (x0,x1), the number of grid points is (609, 255). The dimensions of each dataset are (619, 265), as the simulation uses 5 halo points on either side of the array. The simulation was performed until a non-dimensional time of t=13000, using a 5th order WENO-Z scheme. The simulation is performed at Mach 2, with a similarity solution laminar boundary-layer. The imposed shock is for a pressure ratio between the outlet and inlet of p3/p1 = 1.4, and a shock angle of theta = 32.58. Reynolds number based on displacement thickness of the boundary-layer at the inlet is 950, for further details see the accompanying paper. The available datasets are:

rho_B0 = Density.
rhou0_B0 = Streamwise momentum.
rhou1_B0 = Wall-normal momentum.
rhoE_B0 = Total energy.
x0_B0 = Streamwise coordinate values.
x1_B0 = Wall-normal coordinate values.
Cf_B0 = Wall-normal skin-friction evaluated along the bottom wall.