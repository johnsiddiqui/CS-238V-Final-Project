using Distributions, Statistics, NLsolve, Random

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
    Rb_mean # busbar resistance (ohm)
    Rb_sigma # busbar resistance standard deviation (ohm)
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
    s_count = length(x)
    I_load, Rb_mat, Voc_mat, R0_mat, p_count = params.I_load, params.Rb_mat, params.Voc_mat, params.R0_mat, params.p_count
    for s in 1:s_count
        V_current = x[s]
        V_next = s == s_count ? 0.0 : x[s+1]  # last node is fixed at 0.
        dV = V_current - V_next
        F[s] = sum( (Voc_mat[s, p] - dV) / (R0_mat[s, p] + Rb_mat[s, p]) for p in 1:p_count ) - I_load
    end
end

## Dynamic Simulation
function run_dynamic_simulation(dt,nsteps,arch::ModuleArchitecture,cell::CellModel)
    # Initialize the random resistance matrix for each cell (3x3)
    R0_mat = [rand(Normal(cell.R0_mean, cell.R0_sigma)) for s in 1:arch.s, p in 1:arch.p]
    Rb_mat = [rand(Normal(arch.Rb_mean, arch.Rb_sigma)) for s in 1:arch.s, p in 1:arch.p]

    # SOC history: dimensions (stack s, parallel cell p, time step)
    soc_history = zeros(arch.s, arch.p, nsteps+1)
    soc_history[:, :, 1] = arch.soc_matrix_init

    # Store node voltages at each time step.
    node_voltages = zeros(arch.s, nsteps+1)

    # Voc history for each cell over time (s x p x time step)
    voc_history = zeros(arch.s, arch.p, nsteps+1)
    voc_history[:, :, 1] = build_voc_matrix(cell, arch, arch.soc_matrix_init)

    # Current SOC matrix (3x3)
    soc_matrix = copy(arch.soc_matrix_init)
    
    # Current history for each cell over time (s x p x nsteps)
    current_history = zeros(arch.s, arch.p, nsteps)

    # Initial guess for node voltages
    x_guess = [4.2*arch.s - (i-1)*4.2 for i in 1:arch.s]

    for step in 1:nsteps
        # Build the Voc matrix from the current SOC values.
        Voc_mat = build_voc_matrix(cell,arch,soc_matrix)

        # Solve the nodal equations.
        params = (I_load=arch.I_load, Rb_mat=Rb_mat, Voc_mat=Voc_mat, R0_mat=R0_mat, p_count = arch.p)
        sol = nlsolve(
            (F, x) -> battery_residual!(F, x, params),
            x_guess; 
            method = :trust_region,
            autodiff = :forward
        )
        x_sol = sol.zero # x_sol is a vector of node voltages (length arch.s)

        node_voltages[:, step] = x_sol

        # Compute each cell's current and update SOC.
        Vs = [x_sol... , 0.0]
        for s in 1:arch.s
            dVs = Vs[s] - Vs[s+1] # for stack s, dV = V(s-1) - V(s)
            for p in 1:arch.p
                Icell = (Voc_mat[s, p] - dVs) / (R0_mat[s, p] + Rb_mat[s, p]) # Icell = (Voc - dv) / (R0_cell + Rb)
                current_history[s, p, step] = Icell
                soc_matrix[s, p] -= (Icell * dt) / (Q_cell * 3600) # SOC_new = SOC_old - (Icell*dt)/(Q_cell*3600)
                soc_matrix[s, p] = clamp(soc_matrix[s, p], 0.0, 1.0)
            end
        end

        soc_history[:, :, step+1] = soc_matrix
        voc_history[:, :, step+1] = build_voc_matrix(cell, arch, soc_matrix)
        
        # Use current solution as the next initial guess.
        x_guess = x_sol
    end

    return soc_history, node_voltages, R0_mat, current_history, voc_history
end

function find_max_imbalance(current_history)
    nsteps = size(current_history, 3)
    s_count = size(current_history, 1)

    max_imbalance = -Inf
    max_time = 0
    max_stack = 0

    for t in 1:nsteps
        for s in 1:s_count
            currents = current_history[s, :, t]
            mean_current = mean(currents)
            # Avoid division by zero; if mean_current is 0, we set imbalance to 0.
            imbalance = (mean_current == 0) ? 0.0 : (maximum(currents) - minimum(currents)) / mean_current * 100
            if imbalance > max_imbalance
                max_imbalance = imbalance
                max_time = t
                max_stack = s
            end
        end
    end

    return max_imbalance, max_time, max_stack
end

# record max soc difference
function evalsoc(soc_final)
    mean_soc = mean(soc_final)
    min_soc = minimum(soc_final)
    max_soc = maximum(soc_final)
    soc_imbalance = (max_soc - min_soc)
    return soc_imbalance, mean_soc, min_soc, max_soc
end

# create soc heatmap
function createheatmap(soc_final)
    heatmap(1:size(soc_final,1), 1:size(soc_final,2), soc_final,
        title = "Percent SOC at End of Simulation",
        ylabel = "Parallel Cell Index",
        xlabel = "Series Cell Index",
        ylims = (0.5, p + 0.5),
        xlims = (0.5, s + 0.5),
        aspect_ratio = 1)
end

# create soc percent difference heatmap
function createpercentdiffheatmap(soc_final)
    soc_percentdiff = (soc_final.-mean(soc_final))/mean(soc_final)*100
    heatmap(1:size(soc_final,1), 1:size(soc_final,2), soc_percentdiff,
        title = "SOC Percent Difference at End of Simulation",
        ylabel = "Parallel Cell Index",
        xlabel = "Series Cell Index",
        ylims = (0.5, p + 0.5),
        xlims = (0.5, s + 0.5),
        aspect_ratio = 1,
        color = cgrad(["green", "yellow", "red"]))
end

struct BayesianEstimation
    prior::Beta # from Distributions.jl
    m # number of samples
end

function isfailure(τ,ψ)
    if τ>ψ
        return 1
    else 
        return 0
    end
end

function estimate(alg::BayesianEstimation,data,ψ)
    prior, m = alg.prior, alg.m
    n,m = sum(isfailure(τ,ψ) for τ in data), length(data)
    return Beta(prior.α + n, prior.β + m -n)
end











