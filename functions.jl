using Distributions, Statistics, NLsolve, Random

### Define the Stochastic Model
## Model
struct BranchModel
    n_joints # number of joints in branch
    n_busbars # number of busbars in branch
    r_bus_mean # busbar mean resistances
    r_bus_std # busbar resistance standard deviation
    r_joint_mean # joint mean resistance
    r_joint_std # joint resistance standard deviation
end

## Returns single realization of R_bus
function draw_busbar_resistance(branch::BranchModel)
    dist_bus   = Normal(branch.r_bus_mean, branch.r_bus_std)
    return rand(dist_bus)
end
 
## Returns single realization of R_joint
function draw_joint_resistance(branch::BranchModel)
    dist_joint = Normal(branch.r_joint_mean, branch.r_joint_std)
    return rand(dist_joint)
end

## Returns branch total resistance
function draw_total_resistance(branch::BranchModel)
    total = 0.0
    for steps in 1:branch.n_joints
        total += draw_joint_resistance(branch)
    end 
    for steps in 1:branch.n_busbars
        total += draw_busbar_resistance(branch)
    end
    return total
end

### Circuit Equations for Parallel Busbars
function compute_currents(V, R1, R2)
    I1 = V / R1
    I2 = V / R2
    return I1, I2, (I1 + I2)
end

function current_imbalance(I1, I2; IT = I1 + I2)
    return abs(I1 - I2) / IT * 100
end

### Define Cell Model
## Model
struct CellModel
    ocv_soc # ocv vs soc
    Q_cell # cell capacity
    R0_mean # internal resistance (ohm)
    R0_sigma # internal resistance standard deviation (ohm)
end

struct ModuleArchitecture
    s # series count
    p # parallel count 
    I_load # constant discharge current in A
    Rb # busbar resistor (ohm)
    soc_matrix_init # initial SOCs
end

## Cell Model
function voc(model::CellModel,soci)
    soc = model.ocv_soc[:,1]
    voc = model.ocv_soc[:,2]
    if soci < minimum(soc) || soci > maximum(soc)
        error("soc = $soci is out of bounds [$(minimum(soc))), $(maximum(soc))]")
    end
    
    for i in 1:length(soc)-1
        if soc[i] >= soci >= soc[i+1]
            # Perform linear interpolation:
            return voc[i] + (voc[i+1] - voc[i]) * (soci - soc[i]) / (soc[i+1] - soc[i])
        end
    end
    error("Failed to interpolate: no valid interval found. $soci")
end

## Generate VOC Matrix
function build_voc_matrix(model::CellModel, arch::ModuleArchitecture, soc_matrix)
    Voc_mat = similar(soc_matrix)
    for s in 1:arch.s
        for p in 1:arch.p
            Voc_mat[s, p] = voc(model, soc_matrix[s, p])
        end
    end
    return Voc_mat
end

## Residual function for nodal equations
function battery_residual!(F, x, params)
    V0, V1, V2 = x
    I_load, Rb, Voc_mat, R0_mat = params.I_load, params.Rb, params.Voc_mat, params.R0_mat

    F[1] = (sum( (Voc_mat[1, p]-(V0 - V1)) / (R0_mat[1, p] + Rb) for p in 1:3 ) - I_load)
    F[2] = (sum( (Voc_mat[2, p]-(V1 - V2)) / (R0_mat[2, p] + Rb) for p in 1:3 ) - I_load)
    F[3] = (sum( (Voc_mat[3, p]-(V2 - 0.0)) / (R0_mat[3, p] + Rb) for p in 1:3 ) - I_load)
end

## Dynamic Simulation
function run_dynamic_simulation(dt,nsteps,arch::ModuleArchitecture,cell::CellModel)
    # Initialize the random resistance matrix for each cell (3x3)
    R0_mat = [rand(Normal(cell.R0_mean, cell.R0_sigma)) for s in 1:arch.s, p in 1:arch.p]

    # SOC history: dimensions (stack s, parallel cell p, time step)
    soc_history = zeros(arch.s, arch.p, nsteps+1)
    soc_history[:, :, 1] = arch.soc_matrix_init

    # Store node voltages at each time step.
    node_voltages = zeros(arch.s, nsteps+1)

    # Current SOC matrix (3x3)
    soc_matrix = copy(arch.soc_matrix_init)
    
    # Current history for each cell over time (s x p x nsteps)
    current_history = zeros(arch.s, arch.p, nsteps)

    # Initial guess for node voltages
    x_guess = [12.0, 8.0, 4.0]

    for step in 1:nsteps
        # Build the Voc matrix from the current SOC values.
        Voc_mat = build_voc_matrix(cell,arch,soc_matrix)

        # Solve the nodal equations.
        params = (I_load=arch.I_load, Rb=arch.Rb, Voc_mat=Voc_mat, R0_mat=R0_mat)
        sol = nlsolve(
            (F, x) -> battery_residual!(F, x, params),
            x_guess; 
            method = :trust_region,
            autodiff = :forward
        )
        x_sol = sol.zero
        V0, V1, V2 = x_sol

        node_voltages[:, step] = x_sol

        # Compute each cell's current and update SOC.
        Vs = [V0, V1, V2, 0.0]  # Nodes: [V0, V1, V2, V3=0]
        for s in 1:arch.s
            dVs = Vs[s] - Vs[s+1] # for stack s, dV = V(s-1) - V(s)
            for p in 1:arch.p
                Icell = (Voc_mat[s, p] - dVs) / (R0_mat[s, p] + Rb) # Icell = (Voc - dv) / (R0_cell + Rb)
                current_history[s, p, step] = Icell
                soc_matrix[s, p] -= (Icell * dt) / (Q_cell * 3600) # SOC_new = SOC_old - (Icell*dt)/(Q_cell*3600)
                soc_matrix[s, p] = clamp(soc_matrix[s, p], 0.0, 1.0)
            end
        end

        soc_history[:, :, step+1] = soc_matrix
        # Use current solution as the next initial guess.
        x_guess = x_sol
    end

    return soc_history, node_voltages, R0_mat, current_history
end










