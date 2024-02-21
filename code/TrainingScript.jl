using OrdinaryDiffEq, Optimization, OptimizationOptimisers, SciMLSensitivity, OptimizationOptimJL
using ComponentArrays, Lux, Plots, StableRNGs, JSON, StaticArrays, ArgParse, JLD2, Dates, ThreadPools

function main(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--NAME"
            nargs = 1
            arg_type = String
        "--INPUT"
            nargs = 1
            arg_type = String
        "--SEED"
            nargs = '?'
            arg_type = Int
            default = 1112
        "--WIDTH"
            nargs = '?'
            arg_type = Int
            default = 20
        "--RATE"
            nargs = '?'
            arg_type = Float64
            default = 0.005
        "--DECAY"
            nargs = 4
            arg_type = Float64
            default = [0, 0.1, 9.4, 200] # Cosine Schedule: Min, Step Size, Max, Step Iterations
        "--OUTPUT"
            nargs = 1
            arg_type = String
        "--LOAD"
            action = :store_true
        "--LOAD_PATH"
            nargs = 1
            arg_type = String
        "--LOAD_INDEX"
            nargs = '?'
            arg_type = Int
            default = 1
        "--ADAM"
            action = :store_true
        "--BFGS"
            action = :store_true
        "--WEIGHT_DECAY"
            nargs = '?'
            arg_type = Float64
            default = 0.
        "--PRE_TRAIN"
            action = :store_true
    end

    # ar contains all of the command line parameters
    ar = parse_args(s)
    # random parameter initialization from provided seed
    rng = StableRNG(ar["SEED"]);

    # Where the output is saved to
    out_folder = "$(ar["OUTPUT"][1])/$(ar["NAME"][1])"
    try
        mkdir(out_folder)
    catch
    end

    ### DATA
    infile = JSON.parsefile(ar["INPUT"][1]); # He, a, H, T4
    parameters = infile["meta"]["prams"]
    data = infile["data"]

    losses = Float64[];
    checkpoint_losses = Float64[];

    start_time = now()

    function write_log()
        logfile = open("$(out_folder)/$(ar["NAME"][1])_log.txt", "w")
        write(logfile, """
                NAME: $(ar["NAME"][1])
                SOURCE: $(ar["INPUT"][1])
                TIMESTAMP: $(start_time)
                SEED: $(ar["SEED"])
                WIDTH: $(ar["WIDTH"])
                MAX LEARNING RATE: $(ar["RATE"])
                PARAMS: $(parameters)
                SCHEDULE: ($(join(ar["DECAY"], ",")))
                RECORD LOSS: $(minimum!([1.,], losses)[1])
                __CHECKPOINT LOSSES__\n$(join(checkpoint_losses, "\n"))
                """)
        close(logfile)
    end

    write_log()
    
    function checkpoint(name, losses)
        jldsave("$(out_folder)/$(ar["NAME"][1])_$(name)_checkpoint"; ps)
        lossfile = open("$(out_folder)/$(ar["NAME"][1])_loss.txt","w")
        write(lossfile, join(losses, "\n"))
        close(lossfile)
    end

    for batchname in keys(data)
        data[batchname]["arr"] = [ Float64.(data[batchname][s]) for s in ("xH", "xHe", "T4") ]
        data[batchname]["norm"] = first.(maximum!.([[1.,]], data[batchname]["arr"]));
        data[batchname]["training"] = data[batchname]["arr"] ./ data[batchname]["norm"]
    end

    asteps = infile["meta"]["asteps"];
    aspan = (first(asteps), last(asteps));
    # for normalization
    characteristic_ascale = 1 / (aspan[2] - aspan[1])

    ### NETWORK
    network_u = Lux.Chain(Lux.Dense(7, ar["WIDTH"], tanh), # + H, He, T, a
                            Lux.Dense(ar["WIDTH"], ar["WIDTH"], tanh), 
                            Lux.Dense(ar["WIDTH"], ar["WIDTH"], tanh),
                            Lux.Dense(ar["WIDTH"], ar["WIDTH"], tanh),
                            Lux.Dense(ar["WIDTH"], 3)); # dH, dHe, dT

    p, st = Lux.setup(rng, network_u);

    # system of equations to be integrated
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

    # set up the differential equation solver for the problem
    u = SA[0., 0., 0., 0., 0., 0., 0.] # + H, He, T, a (time)
    ivp = ODEProblem{false}(ude, u, aspan, p);

    function probe_network(p, batchname)
        params = Float64.(data[batchname]["prams"])
        solution = solve(remake(ivp, p=p, u0=SA[1., 1., 1., params[1], params[2], params[3], 0.]), Tsit5(), saveat=asteps, sensealg=QuadratureAdjoint(autojacvec=ZygoteVJP()))
        return [solution[1,:], solution[2,:], solution[3,:]] # xH, xHe, T4
    end


    ##---------- PRE-TRAINING ----------##
    function slope(batch, i)
        if i == length(asteps)
            return [0, 0, 0]
        end

        output = []
        for j in 1:1:3
            push!(output, (data[batch]["training"][j][i + 1] - data[batch]["training"][j][i]) / (asteps[i + 1] - asteps[i]))
        end
        return output
    end

    function ude_d(xH, xHe, T, params, p, t)
        û = network_u([xH, xHe, T, params[1], params[2], params[3], t], p, st)[1] .* characteristic_ascale # Scale to datascale
        return [û[1], û[2], û[3]]
    end

    function probe_network_d(p, batchname)
        params = Float64.(data[batchname]["prams"])
        batchdata = data[batchname]["training"]

        network = [[], [], []]
        for i in 1:1:length(asteps)
            trial = ude_d(batchdata[1][i], batchdata[2][i], batchdata[3][i], params, p, asteps[i])
            push!.(network, trial)
        end
        return network
    end

    function loss_series_d(network, training)
        return sum(abs, (network .- training))
    end

    function loss_batch_d(p, batchname)
        return sum(loss_series_d.(probe_network_d(p, batchname), data[batchname]["derivatives"])) # now needs kmq (or starq?)
    end

    function loss_d(p)
        batches = keys(data)
        return sum(tmap(loss_batch_d, repeat([p], length(batches)), batches))
    end
    ##---------- PRE-TRAINING ----------##

    ### LOSS
    function loss_series(p, network, training)
        return sum(abs, (network .- training))
    end

    function loss_batch(p, batchname)
        return sum(loss_series.([p], probe_network(p, batchname), data[batchname]["training"]))
    end

    function loss(p)
        batches = keys(data)
        return sum(tmap(loss_batch, repeat([p], length(batches)), batches)) + ar["WEIGHT_DECAY"] * sum(abs2, p) # weight decay. ar["WEIGHT_DECAY"] is defaulted to 0
    end

    ### LEARNING RATE SCHEDULE
    decay(x) = cos(x % π/2) # Only get first quadrant
    decay_amounts = decay.(ar["DECAY"][1]:ar["DECAY"][2]:ar["DECAY"][3]);
    learning_rates = Array(ar["RATE"] .* decay_amounts)
    learning_rates = learning_rates[ar["LOAD_INDEX"]:end] # Cutoff early iterations when loading, if you choose

    ### TRAINING
    callback = function (p, l)
        # Save history of losses
        push!(losses, l)
        return false
    end

    optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoForwardDiff());
    function train_network(optf, p0, rates, step_iters)
        optprob = Optimization.OptimizationProblem(optf, p0)
        result = Optimization.solve(optprob, ADAM(rates[1]), callback=callback, maxiters=1);
        ps = result.u

        record_loss = losses[1]
        if ar["PRE_TRAIN"]            
            for batchname in keys(data)
                data[batchname]["derivatives"] = [[], [], []]
                for i in 1:1:length(asteps)
                    push!.(data[batchname]["derivatives"], slope(batchname, i))
                end
            end
            
            optf_d = Optimization.OptimizationFunction((x, p) -> loss_d(x), Optimization.AutoForwardDiff());
            optprob = Optimization.OptimizationProblem(optf_d, ComponentVector{Float64}(ps));
            result = Optimization.solve(optprob, ADAM(rates[1]), callback=callback, maxiters=step_iters);

            record_loss = last(losses)
            push!(checkpoint_losses, record_loss)
            ps = result.u

            checkpoint("Pre-training", losses)
            write_log()
        end
        if ar["ADAM"]
            for rate in rates
                optprob = Optimization.OptimizationProblem(optf, ps)
                result = Optimization.solve(optprob, ADAM(rate), callback=callback, maxiters=step_iters);

                record_loss = last(losses)
                push!(checkpoint_losses, record_loss)
                ps = result.u

                checkpoint("$(length(losses))", losses)
                write_log()
            end
        end
        if ar["BFGS"]
            for rate in (rates[1]*10, rates[1]*3, rates[1], rates[1]/3, rates[1]/10)
                optprob = Optimization.OptimizationProblem(optf, ps)
                result = Optimization.solve(optprob,
                                            Optim.BFGS(initial_stepnorm=rate),
                                            callback=callback,
                                            allow_f_increases = false);
                
                record_loss = last(losses)
                push!(checkpoint_losses, record_loss)
                ps = result.u

                checkpoint("BFGS$(rate)", losses)
                write_log()
            end
        end

        checkpoint("FinalNetworkParameters", losses)
    end

    p = ComponentVector{Float64}(p)

    # Choice to load in partly trained network
    if ar["LOAD"]
        p = jldopen(ar["LOAD_PATH"][1])["ps"]
    else
    end

    # Do the training
    train_network(optf, p, learning_rates, Int(ar["DECAY"][4]));

    write_log()
end

main(ARGS)