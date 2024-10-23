using JLD2, Plots, Lux, OrdinaryDiffEq, SciMLSensitivity, JSON, ComponentArrays, StableRNGs, StaticArrays, Plots.PlotMeasures, ArgParse

function main(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--NAME"
            nargs = 1
            arg_type = String
        "--INPUT"
            nargs = 1
            arg_type = String
        "--DATA"
            nargs = 1
            arg_type = String
        "--WIDTH"
            nargs = '?'
            arg_type = Int
            default = 30
        "--PARAMS"
            nargs = 3
            arg_type = Float64
            default = [0.0457, 0.249, 2.725] # Planck 2018 best fits
    end

    ar = parse_args(s)

    load_path = ar["INPUT"][1]
    data_path = ar["DATA"][1]
    name = "ps"
    width = ar["WIDTH"]
    params = ar["PARAMS"]

    rng = StableRNG(1111)

    infile = JSON.parsefile(data_path); # He, a, H, T4

    asteps = infile["meta"]["asteps"];
    aspan = (first(asteps), last(asteps));
    characteristic_ascale = 1 / (aspan[2] - aspan[1])

    ### CREATE THE NETWORK
    network_u = Lux.Chain(Lux.Dense(7, width, tanh), # + H, He, T, a
                            Lux.Dense(width, width, tanh), 
                            Lux.Dense(width, width, tanh), 
                            Lux.Dense(width, width, tanh), 
                            Lux.Dense(width, 3)); # dH, dHe, dT

    p, st = Lux.setup(rng, network_u);

    ### FUNCTION TO PROBE THE NETWORK
    function ude(u, p, t)
        û = network_u(SA[u[1], u[2], u[3], u[4], u[5], u[6], t], p, st)[1] .* characteristic_ascale  # Scale to datascale
        du1 = u[1] + û[1]
        du2 = u[2] + û[2]
        du3 = u[3] + û[3]
        du4 = 0
        du5 = 0
        du6 = 0
        du7 = 0
        return SA[du1, du2, du3, du4, du5, du6, du7]
    end

    u = SA[0., 0., 0., 0., 0., 0., 0.] # + H, He, T, a (time)
    ivp = ODEProblem{false}(ude, u, aspan, p);

    function probe_network(p, batchname)
        params = Float64.(data[batchname]["prams"])
        solution = solve(remake(ivp, p=p, u0=SA[1., 1., 1., params[1], params[2], params[3], 0.]), Tsit5(), saveat=asteps, sensealg=QuadratureAdjoint(autojacvec=ZygoteVJP()))
        return [solution[1,:], solution[2,:], solution[3,:]] # xH, xHe, T4
    end

    # load in the parameters from training
    p_load = jldopen(load_path)[name]

    # Make and save the plot
    network_data = probe_network(p_load, 64)
    plt = plot(asteps, network_data[1], label = "Network Hydrogen",
                        title="$(ar["NAME"][1])", xlabel="Scale Factor", left_margin=5mm);
    plot!(plt, asteps, network_data[2], label = "Network Helium");
    plot!(plt, asteps, network_data[3], label = "Network Temperature");
    savefig("$(ar["NAME"][1]).png")
end

main(ARGS)