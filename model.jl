include("functions.jl")

## System Parameters
r_bus_mean   = 1.0e-3   # 1 mΩ, for example
r_bus_std    = 0.2e-3   # ±0.2 mΩ std dev
r_joint_mean = 0.3e-3
r_joint_std  = 0.05e-3
V = 100.0  # 100 V, for example
n_joints = 1
n_busbars = 2

## Model
ocv_soc = [1 4.19;0.9 4.09;0.8 4.02;0.7 3.92; 0.6 3.81;0.5 3.71;0.4 3.64;0.3 3.56;0.2 3.47;0.1 3.26];
R0_mean = 15/1000 #ohm
R0_sigma = 2/1000 #ohm
Q_cell = 2.5 #Ah

## Architecture 
soc_matrix_init = [
    0.90  0.85  0.88;
    0.80  0.75  0.78;
    0.95  0.92  0.90
]
s = 3
p = 3
Rb = 0.0005           # Busbar resistor (ohm)
I_load = 10        # Constant discharge current in A
dt = 1.0             # Time step (seconds)
nsteps = 1500          # Number of simulation steps

## Define
currentModel = CellModel(ocv_soc,Q_cell,R0_mean,R0_sigma)
currentArchitecture = ModuleArchitecture(s,p,I_load,Rb,soc_matrix_init)

## Run
soc_history, node_voltages, R0_mat, current_history = run_dynamic_simulation(dt, nsteps, currentArchitecture, currentModel)

println("Final SOC matrix after $nsteps steps:")
println(round.(soc_history[:, :, end], digits=4))

println("\nFinal node voltages:")
println(round.(node_voltages[:, end], digits=3))

println("\nStochastic R0 values (ohm) for each cell:")
println(round.(R0_mat, digits=4))
