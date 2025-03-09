include("functions.jl")
using Plots

## Model
ocv_soc = [1 4.19;0.9 4.09;0.8 4.02;0.7 3.92; 0.6 3.81;0.5 3.71;0.4 3.64;0.3 3.56;0.2 3.47;0.1 3.26];
R0_mean = 15/1000 #ohm
R0_sigma = 4/1000 #ohm
Q_cell = 2.5 #Ah

## Architecture 
s = 22 #assuming 5 modules or 110*3.6 or 396 nominal voltage
p = 8 
init_soc = 0.9
soc_matrix_init = fill(init_soc,s,p)
Rb_mean = 0.0005 # busbar resistance (ohm)
Rb_sigma = 0.0001 # busbar resistance standard deviation (ohm)
I_load = 2.5*Q_cell*p       # Constant discharge current in A
dt = 1.0          # Time step (seconds)
nsteps = 1000     # Number of simulation steps

## Define
currentModel = CellModel(ocv_soc,Q_cell,R0_mean,R0_sigma)
currentArchitecture = ModuleArchitecture(s,p,I_load,Rb_mean,Rb_sigma,soc_matrix_init)

## Run
soc_history, node_voltages, R0_mat, current_history, voc_history = run_dynamic_simulation(dt, nsteps, currentArchitecture, currentModel)

println("Final SOC matrix after $nsteps steps:")
println(round.(soc_history[:, :, end], digits=4))

println("\nFinal VOC after $nsteps steps:")
println(round.(voc_history[:, :, end], digits=4))

println("\nStochastic R0 values (ohm) for each cell:")
println(round.(R0_mat, digits=4))

max_imb, t_step, stack_idx = find_max_imbalance(current_history)
println("Max imbalance: $max_imb% at time step $t_step in stack $stack_idx")