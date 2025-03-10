include("functions.jl")
using Plots

## Model
ocv_soc = [1 4.19;0.9 4.09;0.8 4.02;0.7 3.92; 0.6 3.81;0.5 3.71;0.4 3.64;0.3 3.56;0.2 3.47;0.1 3.26];
R0_mean = 15/1000 #ohm
R0_sigma = 2/1000 #ohm
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

## Simulation Parameters
m = 100
results_mat = zeros(m,2)

## Run
for index in 1:m
    completionPercent = round(index/m*100,digits=2)
    soc_history, node_voltages, R0_mat, current_history, voc_history = run_dynamic_simulation(dt, nsteps, currentArchitecture, currentModel)

    soc_final = soc_history[:, :, end]
    soc_imbalance, mean_soc, min_soc, max_soc = evalsoc(soc_final)
    percent_mean_soc = round(mean_soc*100,digits=2)
    percent_imbalance = round(soc_imbalance*100,digits=2)
    println("Max percent imbalance: $percent_imbalance%")
    println("Mean SOC: $percent_mean_soc%")
    println("Percent Complete: $completionPercent%\n")
    
    results_mat[index,1] = soc_imbalance
    results_mat[index,2] = mean_soc
    #createheatmap(soc_final)
    #createpercentdiffheatmap(soc_final)
end

## Estimate Failure Prob
data = results_mat[:,1]
ψ = 0.10
posterior = estimate(BayesianEstimation(Beta(1,1),m),data,ψ)
confidence = cdf(posterior,0.01)
bound = quantile(posterior,0.95)

# Fit a Normal distribution to the data using MLE.
fitted_normal = fit(Normal, data)
println("Fitted Normal Distribution: μ = $(fitted_normal.μ), σ = $(fitted_normal.σ)")

# Plot the histogram of the data (normalized to form a PDF)
histogram(data, normalize=true, xlabel="Percent SOC Imbalance",
    label="Normalized Data",nbins=30,ylabel="Density",alpha=0.5)

# Plot the fitted PDF on top of the histogram
x = range(minimum(data), stop=maximum(data), length=200)
plot!(x, pdf.(fitted_normal, x), label="Fitted Normal PDF", lw=2, color="red")



