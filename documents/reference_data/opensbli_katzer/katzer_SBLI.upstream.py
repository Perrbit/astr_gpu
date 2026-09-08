#!/usr/bin/env python
from opensbli import *
import copy
from opensbli.utilities.katzer_init import Initialise_Katzer
from opensbli.utilities.helperfunctions import substitute_simulation_parameters

simulation_parameters = {
'gama'      :   '1.4',
'Minf'      :   '2.0',
'Pr'        :   '0.72',
'Re'        :   '950.0',
'Twall'     :   '1.67619431',
'dt'        :   '0.04',
'niter'     :   '250000',
'block0np0'     :   '500',
'block0np1'     :   '250',
'Delta0block0'      :   '400.0/(block0np0-1)',
'Delta1block0'      :   '115.0/(block0np1-1)',
'SuthT'     :   '110.4',
'RefT'      :   '288.0',
'eps'       :   '1e-15',
'TENO_CT'       :   '1e-5',
'Lx1'       :   '115.0',
'by'        :   '5.0',
'teno_a1'       :   '10.5',
'teno_a2'       :   '4.5',
'epsilon'       :   '1.0e-30',
'lambda0_TVD'       : 'dt/Delta0block0',
'lambda1_TVD'       : 'dt/Delta1block0',
'inv_rfact0_block0' : '1.0/Delta0block0',
'inv_rfact1_block0' : '1.0/Delta1block0',
}

ndim = 2
# Direct application of shock-capturing scheme, otherwise central scheme with filter-step example
weno = True
teno = False
TVD = False
# Instatiate equation classes
eq = EinsteinEquation()
constants = ["Re", "Pr", "gama", "Minf", "SuthT", "RefT"]
# Define the problem
if weno or teno:
    if weno:
        sc1 = "**{\'scheme\':\'Weno\'}"
    else:
        sc1 = "**{\'scheme\':\'Teno\'}"
    # Define the compresible Navier-Stokes equations in Einstein notation.
    mass = "Eq(Der(rho,t), - Conservative(rhou_j,x_j,%s))" % sc1
    momentum = "Eq(Der(rhou_i,t) , -Conservative(rhou_i*u_j + KD(_i,_j)*p,x_j , %s) + Der(tau_i_j,x_j) )" % sc1
    energy = "Eq(Der(rhoE,t), - Conservative((p+rhoE)*u_j,x_j, %s) - Der(q_j,x_j) + Der(u_i*tau_i_j ,x_j) )" % sc1
    stress_tensor = "Eq(tau_i_j, (mu/Re)*(Der(u_i,x_j)+ Der(u_j,x_i) - (2/3)* KD(_i,_j)* Der(u_k,x_k)))"
    heat_flux = "Eq(q_j, (-mu/((gama-1)*Minf*Minf*Pr*Re))*Der(T,x_j))"
    # Substitutions
    substitutions = [stress_tensor, heat_flux]
    mass = eq.expand(mass, ndim, "x", substitutions, constants)
    momentum = eq.expand(momentum, ndim, "x", substitutions, constants)
    energy = eq.expand(energy, ndim, "x", substitutions, constants)
else:
    NS = NS_Split('KGP', ndim, constants, coordinate_symbol="x", conservative=True, viscosity='dynamic', energy_formulation='enthalpy', debug=False)
    mass, momentum, energy = NS.mass, NS.momentum, NS.energy


# Expand the simulation equations, for this create a simulation equations class
simulation_eq = SimulationEquations()
simulation_eq.add_equations(mass)
simulation_eq.add_equations(momentum)
simulation_eq.add_equations(energy)

# Formulas for the variables used in the equations
velocity = "Eq(u_i, rhou_i/rho)"
pressure = "Eq(p, (gama-1)*(rhoE - rho*(1/2)*(KD(_i,_j)*u_i*u_j)))"
speed_of_sound = "Eq(a, (gama*p/rho)**0.5)"
temperature = "Eq(T, p*gama*Minf*Minf/(rho))"
viscosity = "Eq(mu, (T**(1.5)*(1.0+SuthT/RefT)/(T+SuthT/RefT)))"

constituent_eqns = [velocity, pressure, speed_of_sound, temperature, viscosity]
constituent = ConstituentRelations()
# Expand the constituent relations
for i, CR in enumerate(constituent_eqns):
    constituent_eqns[i] = eq.expand(CR, ndim, "x", [], constants)

for eqn in constituent_eqns:
    constituent.add_equations(eqn)

# Create a simulationblock
block = SimulationBlock(ndim, block_number=0)

# Create metrics before the scheme selection
metriceq = MetricsEquation()
metriceq.generate_transformations(ndim, "x", [(False, False), (True, False)], 2)
# Grid is stretched normal to the wall
simulation_eq.apply_metrics(metriceq)

# Adaptive TENO with modified Ducros sensor
# SS = ShockSensor()
# shock_sensor, sensor_array = SS.ducros_equations(block, coordinate_symbol, metriceq)
# Add shock Ducros sensor to constituent relations
# constituent.add_equations(shock_sensor)
store_sensor = True
# Spatial scheme
schemes = {}
if weno:
    Avg = SimpleAverage([0, 1])
    LF = LFWeno(order=5, formulation='Z', averaging=Avg, flux_type='LLF')
    #LF = HLLCWeno(order=5, formulation='Z', averaging=Avg, flux_type='HLLC-LM')
    # Add to schemes
    schemes[LF.name] = LF
elif teno:
    Avg = SimpleAverage([0, 1])
    LF = LFTeno(order=5, averaging=Avg, flux_type='LLF')
    #LF = HLLCTeno(order=6, averaging=Avg, flux_type='HLLC-LM')
    # Add to schemes
    schemes[LF.name] = LF   
# Central scheme 
fns = 'u0 u1 T'
cent = StoreSome(4, fns)
schemes[cent.name] = cent

# Time-stepping
rk = RungeKuttaLS(3, formulation='SSP')
schemes[rk.name] = rk
block.set_discretisation_schemes(schemes)

local_dict = {"block": block, "GridVariable": GridVariable, "DataObject": DataObject}
x_loc = parse_expr("Eq(GridVariable(x0), block.deltas[0]*block.grid_indexes[0])", local_dict=local_dict)
rho = parse_expr("Eq(DataObject(rho), d)", local_dict=local_dict)
rhou0 = parse_expr("Eq(DataObject(rhou0), d*u0)", local_dict=local_dict)
rhou1 = parse_expr("Eq(DataObject(rhou1), d*u1)", local_dict=local_dict)
rhoE = parse_expr("Eq(DataObject(rhoE), p/(gama-1) + 0.5* d *(u0**2+u1**2))", local_dict=local_dict)

boundaries = [[0, 0] for t in range(ndim)]
# Left pressure extrapolation at x= 0, inlet conditions
direction = 0
side = 0
boundaries[direction][side] = InletPressureExtrapolateBC(direction, side)
# Right extrapolation at outlet
direction = 0
side = 1
boundaries[direction][side] = ExtrapolationBC(direction, side, order=0)
# Bottom no-slip isothermal wall
direction = 1
side = 0
wall_const = ["Minf", "Twall"]
for con in wall_const:
    local_dict[con] = ConstantObject(con)
# Isothermal wall condition
rhoE_wall = parse_expr("Eq(DataObject(rhoE), DataObject(rho)*Twall/(gama*(gama-1.0)*Minf**2.0))", local_dict=local_dict)
wall_eqns = [rhoE_wall]
boundaries[direction][side] = IsothermalWallBC(direction, 0, wall_eqns)
# Top dirichlet shock generator condition
direction = 1
side = 1
rho = parse_expr("Eq(DataObject(rho), Piecewise((1.129734572, (x0)>40.0), (1.00000596004, True)))", local_dict=local_dict)
rhou0 = parse_expr("Eq(DataObject(rhou0), Piecewise((1.0921171, (x0)>40.0), (1.00000268202, True)))", local_dict=local_dict)
rhou1 = parse_expr("Eq(DataObject(rhou1), Piecewise((-0.058866065, (x0)>40.0), (0.00565001630205, True)))", local_dict=local_dict)
rhoE = parse_expr("Eq(DataObject(rhoE), Piecewise((1.0590824, (x0)>40.0), (0.94644428042, True)))", local_dict=local_dict)

upper_eqns = [x_loc, rho, rhou0, rhou1, rhoE]
boundaries[direction][side] = DirichletBC(direction, side, upper_eqns)

block.set_block_boundaries(boundaries)

# Perform initial condition
# Reynolds number, Mach number and free-stream temperature for the initial profile
Re, xMach, Tinf = 950.0, 2.0, 288.0
## Ensure the grid size passed to the initialisation routine matches the grid sizes used in the simulation parameters
polynomial_directions = [(False, DataObject('x0')), (True, DataObject('x1'))]
n_poly_coefficients = 50
grid_const = ["Lx1", "by"]
for con in grid_const:
    local_dict[con] = ConstantObject(con)
gridx0 = parse_expr("Eq(DataObject(x0), block.deltas[0]*block.grid_indexes[0])", local_dict=local_dict)
gridx1 = parse_expr("Eq(DataObject(x1), Lx1*sinh(by*block.deltas[1]*block.grid_indexes[1]/Lx1)/sinh(by))", local_dict=local_dict)
coordinate_evaluation = [gridx0, gridx1]
initial = Initialise_Katzer(polynomial_directions, n_poly_coefficients,  Re, xMach, Tinf, coordinate_evaluation)

kwargs = {'iotype': "Write"}
h5 = iohdf5(**kwargs)
h5.add_arrays(simulation_eq.time_advance_arrays)
h5.add_arrays([DataObject('x0'), DataObject('x1'), DataObject('D11')])#, DataObject('kappa'), DataObject('WENO_filter')])
block.setio(h5)

# Set equations on the block and discretise
block.set_equations([constituent, simulation_eq, initial, metriceq])
# WENO/TVD filter if not using direct application of WENO/TENO
if not weno and not teno:
    if TVD:
        TVD_filter = TVDFilter(block, airfoil=False)
        block.set_equations(TVD_filter.equation_classes)
    else:
        WF = WENOFilter(block, order=5, formulation='Z', flux_type='LLF', airfoil=False, metrics=metriceq)
        block.set_equations(WF.equation_classes)  


block.discretise()

alg = TraditionalAlgorithmRK(block)
mixed_precision = False
if mixed_precision:
    SimulationDataType.set_datatype(FloatC)
    # Define custom precision options
    custom_arrays = [block.location_dataset(dset) for dset in ['rho', 'p']]
    # Create the dictionary of mixed precision options
    mixed_precision_config = {
    'q_vector' : ([], Double),
    'RK_arrays' : ([], Double),
    # 'casting' : 'explicit',
    # 'residuals' : ([], FloatC),
    # 'wk_arrays' : ([], FloatC),
    # 'custom'   : (custom_arrays, FloatC),
    }
    OPSC(alg, mixed_precision_config=mixed_precision_config)
else:
    SimulationDataType.set_datatype(Double)
    OPSC(alg)
# Add the simulation constants to the OPS C code
substitute_simulation_parameters(simulation_parameters.keys(), simulation_parameters.values())
print_iteration_ops(NaN_check='rho')
