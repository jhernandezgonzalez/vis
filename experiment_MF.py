import numpy as np
import matplotlib.pyplot as plt
import resource


def cpu_time():
    return resource.getrusage(resource.RUSAGE_SELF)[0]


def scenario_with_many_samplings_and_rates_i(problem_gen, num_repeats, num_samples, num_measures, algorithms,
                                             rates_i, rnd_seed=7):
    np.random.seed(rnd_seed)

    real_vals = np.zeros((1, num_repeats, num_measures))
    estimates = {}
    variances = {}
    theo_variances = {}

    print("+++++++++++++++++++++ Generate Problem +++++++++++++++++++++")
    start = cpu_time()
    f = problem_gen()
    f.toExp()
    end = cpu_time()
    print("[time] Model generation:", end - start)

    start = cpu_time()
    real_vals[:, :] = np.log(np.sum(f.joint().t))
    end = cpu_time()
    print("[time] Z real value:", end - start)

    '''start = cpu_time()
    h_s = np.zeros(len(f.factors))
    for i_a, a in enumerate(f.factors):
        h_s[i_a] = a.entropy() / np.log(np.prod(a.dims()))
    #print(h_s)
    end = cpu_time()
    print("Mean entropy:", np.mean(h_s), "+-", np.std(h_s))
    print("[time] Mean norm. entropy:", end - start)'''

    print("++++++++++++++++++++++++++++ Testing ++++++++++++++++++++++++++++")

    for alg in algorithms:
        print("Testing algorithm:",alg.name)
        alg.set_real_value(real_vals[0, 0, 0])
        start = cpu_time()
        alg.set_problem(f.copy()) #alg.project_onto_q() set_problem now includes projection
        end = cpu_time()
        print("[time] Projection:", end - start)

        if alg.name == "VIS-Sp-I-est":
            for r in rates_i:
                estimates[alg.name + "-" + str(r)] = np.zeros((1, num_repeats, num_measures))
                variances[alg.name + "-" + str(r)] = np.zeros((1, num_repeats, num_measures))
                theo_variances[alg.name + "-" + str(r)] = np.zeros((1, num_repeats, num_measures))
            start = cpu_time()
            #theo_variances[alg.name][rp, :, :] = alg.theoretical_variance(num_samples, num_measures)
            for rs in range(num_repeats):
                for r in rates_i:
                    alg.set_rate_i(r)
                    estimates[alg.name + "-" + str(r)][0, rs, :], variances[alg.name + "-" + str(r)][0, rs, :] = \
                        alg.estimate(num_samples, num_measures)
                alg.clean_sample_reserve()
            end = cpu_time()
            print("[time] Z estimation:", (end - start)/num_repeats, "(av.)")
        else:
            estimates[alg.name] = np.zeros((1, num_repeats, num_measures))
            variances[alg.name] = np.zeros((1, num_repeats, num_measures))
            theo_variances[alg.name] = np.zeros((1, num_repeats, num_measures))
            start = cpu_time()
            #theo_variances[alg.name][rp, :, :] = alg.theoretical_variance(num_samples, num_measures)
            for rs in range(num_repeats):
                estimates[alg.name][0, rs, :], variances[alg.name][0, rs, :] = alg.estimate(num_samples, num_measures)
            end = cpu_time()
            print("[time] Z estimation:", (end - start)/num_repeats, "(av.)")
    return real_vals, estimates, variances, theo_variances



def scenario(problem_gen, num_problems, num_repeats, num_samples, num_measures, algorithms, rnd_seed=7):
    np.random.seed(rnd_seed)

    real_vals = np.zeros((num_problems, num_repeats, num_measures))
    estimates = {}
    variances = {}
    times = {}
    theo_variances = {}
    for ac, alg in enumerate(algorithms):
        #alg.set_arguments(**arguments[ac])
        estimates[alg.name] = np.zeros((num_problems, num_repeats, num_measures))
        variances[alg.name] = np.zeros((num_problems, num_repeats, num_measures))
        times[alg.name] = np.zeros(num_problems)
        theo_variances[alg.name] = np.zeros((num_problems, num_repeats, num_measures))

    f_problems = []
    for rp in range(num_problems):
        print("+++++++++++++++++++++ Generate Problem no.", rp+1, "+++++++++++++++++++++")
        start = cpu_time()
        f_problems.append( problem_gen() )
        f_problems[rp].toExp()
        end = cpu_time()
        print("[time] Model generation:", end - start)

        start = cpu_time()
        real_vals[rp, :, :] = np.log(np.sum(f_problems[rp].joint().t))
        end = cpu_time()
        print("[time] Z real value:", end - start)

        '''start = cpu_time()
        h_s = np.zeros(len(f_problems[rp].factors))
        for i_a, a in enumerate(f_problems[rp].factors):
            h_s[i_a] = a.entropy() / np.log(np.prod(a.dims()))
        #print(h_s)
        end = cpu_time()
        print("Mean entropy:", np.mean(h_s), "+-", np.std(h_s))
        print("[time] Mean norm. entropy:", end - start)'''

    # for rp in tqdm(np.arange(num_problems), desc='Different problems', leave=False):
    for rp in range(num_problems):
        print("+++++++++++++++++++++ Testing for Problem no.", rp+1, "+++++++++++++++++++++")

        for alg in algorithms:
            print("Testing algorithm:",alg.name)
            alg.set_real_value(real_vals[rp, 0, 0])
            start = cpu_time()
            alg.set_problem(f_problems[rp].copy()) #alg.project_onto_q()  set_problem now includes projection
            end = cpu_time()
            times[alg.name][rp] = end - start
            print("[time] Projection:", times[alg.name][rp])

            start = cpu_time()
            #theo_variances[alg.name][rp, :, :] = alg.theoretical_variance(num_samples, num_measures)
            for rs in range(num_repeats):
                estimates[alg.name][rp, rs, :], variances[alg.name][rp, rs, :] = alg.estimate(num_samples, num_measures)
            end = cpu_time()
            print("[time] Z estimation:", (end - start)/num_repeats, "(av.)")
    return real_vals, estimates, variances, theo_variances, times


def scenario_time_consumption(problem_gen, num_problems, algorithms, rnd_seed=7):
    np.random.seed(rnd_seed)

    proj_time = {}
    for ac, alg in enumerate(algorithms):
        proj_time[alg.name] = np.zeros(num_problems)

    f_problems = []
    for rp in range(num_problems):
        print("+++++++++++++++++++++ Generate Problem no.", rp+1, "+++++++++++++++++++++")
        start = cpu_time()
        f_problems.append( problem_gen() )
        f_problems[rp].toExp()
        end = cpu_time()
        print("[time] Model generation:", end - start)

    for rp in range(num_problems):
        print("+++++++++++++++++++++ Testing for Problem no.", rp+1, "+++++++++++++++++++++")

        for alg in algorithms:
            print("Testing algorithm:",alg.name)
            start = cpu_time()
            alg.set_problem(f_problems[rp].copy())
            end = cpu_time()
            proj_time[alg.name][rp] = end - start
            print("[time] Projection:", proj_time[alg.name][rp])

    return proj_time



def plot_results(real_vals, estimates, variances, theo_variances, num_samples, num_measures,
                 logscale=None, do_not_show=[], save_figure=False, limits_without_smallest=False, filename="aux.png"):
    #col = 0

    xaxis = np.arange(0,num_samples+1,num_samples//num_measures)[1:]

    fig, ax = plt.subplots(2, 2, figsize=(20, 16)) # 10,3

    to_show_algs = [ alg_name for alg_name in estimates if alg_name not in do_not_show]
    colors = ["C" + s for s in list(map(str, np.arange(len(to_show_algs))))]
    '''lnames = list(estimates.keys())
    lnames.remove("sMC")
    lnames.remove("DEst")
    lnames.remove("VIS-Sp-I-0.25")
    for col, alg_name in enumerate(lnames):'''
    min_val = [[],[],[],[]]
    for col, alg_name in enumerate(to_show_algs):
        point_rel_error = np.abs(estimates[alg_name] - real_vals)/real_vals
        ax[0,0].plot(xaxis, np.median(point_rel_error, axis=(0, 1)), label=alg_name, color=colors[col])
        ax[1,0].plot(xaxis, np.mean(point_rel_error, axis=(0, 1)), label=alg_name, color=colors[col])

        min_val[0].append(np.min(np.median(point_rel_error, axis=(0, 1))))
        min_val[1].append(np.min(np.mean(point_rel_error, axis=(0, 1))))

        high_pct_error = np.percentile(point_rel_error, 75, axis=(0, 1))
        low_pct_error = np.percentile(point_rel_error, 25, axis=(0, 1))
        ax[0,0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)
        ax[0,0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)

        ax[1,0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)
        ax[1,0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)

        ax[0,1].plot(xaxis, np.median(variances[alg_name], axis=(0, 1)), label=alg_name, color=colors[col])
        ax[1,1].plot(xaxis, np.mean(variances[alg_name], axis=(0, 1)), label=alg_name, color=colors[col])
        #ax[1].plot(xaxis, np.mean(theo_variances[alg_name], axis=(0, 1)), label=alg_name, alpha=0.7, color=colors[col])

        min_val[2].append(np.min(np.median(variances[alg_name], axis=(0, 1))))
        min_val[3].append(np.min(np.mean(variances[alg_name], axis=(0, 1))))

    ax[0,0].set_title("Estimation")
    ax[0,0].set_xlabel("Num. samples")
    ax[0,0].set_ylabel("Median relative error")
    ax[0,0].legend()

    ax[1,0].set_xlabel("Num. samples")
    ax[1,0].set_ylabel("Mean relative error")

    ax[0,1].set_title("Variance")
    ax[0,1].set_xlabel("Num. samples")
    ax[0,1].set_ylabel("Median variance")

    ax[1,1].set_xlabel("Num. samples")
    ax[1,1].set_ylabel("Mean variance")

    if limits_without_smallest:
        min_val[2].sort()
        x1, x2, y1, y2 = ax[0,1].axis()
        ax[0,1].axis((x1, x2, min_val[2][1]/2, y2))

        min_val[3].sort()
        x1, x2, y1, y2 = ax[1,1].axis()
        ax[1,1].axis((x1, x2, min_val[3][1]/2, y2))

    if (logscale is not None):
        for ls in logscale:
            ax[ls].set_yscale('log')
            ax[ls].set_ylabel("Log-" + ax[ls].get_ylabel())

    plt.show()
    if save_figure:
        fig.savefig(filename,dpi=fig.dpi, bbox_inches='tight')


def plot_times(proj_times, logscale=False, xticks=None, do_not_show=[], save_figure=False, filename="aux.png"):
    xaxis = np.arange(len(proj_times))+1

    fig, ax = plt.subplots(1, 1, figsize=(4, 3)) # 10,3

    to_show_algs = [ alg_name for alg_name in proj_times[0] if alg_name not in do_not_show]
    colors = ["C" + s for s in list(map(str, np.arange(len(to_show_algs))))]

    for col, alg_name in enumerate(to_show_algs):
        vals = [np.mean(proj_times[i_d][alg_name]) for i_d in np.arange(len(proj_times))]
        ax.plot(xaxis, vals, label=alg_name, color=colors[col])

    ax.set_title("Time consumption")
    ax.set_xlabel("delta")
    if xticks is not None:
        ax.set_xticks(xaxis)
        ax.set_xticklabels(xticks)
    ax.set_ylabel("time")
    ax.legend()

    if logscale:
        ax.set_yscale('log')
        ax.set_ylabel("log-" + ax.get_ylabel())

    plt.show()
    if save_figure:
        fig.savefig(filename,dpi=fig.dpi, bbox_inches='tight')



def subplot_results(real_vals, estimates, variances, theo_variances, num_samples, num_measures, measures_to_show=None,
                 logscale=None, do_not_show=[], save_figure=False, limits_without_smallest=False, filename="aux.png"):
    #col = 0

    if measures_to_show is None:
        measures_to_show = range(num_measures)
    else:
        measures_to_show = range(measures_to_show)

    xaxis = np.arange(0,num_samples+1,num_samples//num_measures)[1:]
    xaxis = xaxis[measures_to_show]

    fig, ax = plt.subplots(2, 2, figsize=(20, 16)) # 10,3

    to_show_algs = [ alg_name for alg_name in estimates if alg_name not in do_not_show]
    colors = ["C" + s for s in list(map(str, np.arange(len(to_show_algs))))]
    '''lnames = list(estimates.keys())
    lnames.remove("sMC")
    lnames.remove("DEst")
    lnames.remove("VIS-Sp-I-0.25")
    for col, alg_name in enumerate(lnames):'''
    min_val = [[],[],[],[]]
    for col, alg_name in enumerate(to_show_algs):
        point_rel_error = np.abs(estimates[alg_name] - real_vals)/real_vals
        ax[0,0].plot(xaxis, np.median(point_rel_error, axis=(0, 1))[measures_to_show], label=alg_name, color=colors[col])
        ax[1,0].plot(xaxis, np.mean(point_rel_error, axis=(0, 1))[measures_to_show], label=alg_name, color=colors[col])

        min_val[0].append(np.min(np.median(point_rel_error, axis=(0, 1))[measures_to_show]))
        min_val[1].append(np.min(np.mean(point_rel_error, axis=(0, 1))[measures_to_show]))

        high_pct_error = np.percentile(point_rel_error, 75, axis=(0, 1))[measures_to_show]
        low_pct_error = np.percentile(point_rel_error, 25, axis=(0, 1))[measures_to_show]
        ax[0,0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)
        ax[0,0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)

        ax[1,0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)
        ax[1,0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)

        ax[0,0].set_title("Estimation")
        ax[0,0].set_xlabel("Num. samples")
        ax[0,0].set_ylabel("Median relative error")
        ax[0,0].legend()

        ax[1,0].set_xlabel("Num. samples")
        ax[1,0].set_ylabel("Mean relative error")

        ax[0,1].plot(xaxis, np.median(variances[alg_name], axis=(0, 1))[measures_to_show], label=alg_name, color=colors[col])
        ax[1,1].plot(xaxis, np.mean(variances[alg_name], axis=(0, 1))[measures_to_show], label=alg_name, color=colors[col])
        #ax[1].plot(xaxis, np.mean(theo_variances[alg_name], axis=(0, 1)), label=alg_name, alpha=0.7, color=colors[col])

        min_val[2].append(np.min(np.median(variances[alg_name], axis=(0, 1))[measures_to_show]))
        min_val[3].append(np.min(np.mean(variances[alg_name], axis=(0, 1))[measures_to_show]))

        ax[0,1].set_title("Variance")
        ax[0,1].set_xlabel("Num. samples")
        ax[0,1].set_ylabel("Median variance")

        ax[1,1].set_xlabel("Num. samples")
        ax[1,1].set_ylabel("Mean variance")

    if limits_without_smallest:
        min_val[2].sort()
        x1, x2, y1, y2 = ax[0,1].axis()
        ax[0,1].axis((x1, x2, min_val[2][1]/2, y2))

        min_val[3].sort()
        x1, x2, y1, y2 = ax[1,1].axis()
        ax[1,1].axis((x1, x2, min_val[3][1]/2, y2))

    if (logscale is not None):
        for ls in logscale:
            ax[ls].set_yscale('log')
            ax[ls].set_ylabel("Log-" + ax[ls].get_ylabel())

    plt.show()
    if save_figure:
        fig.savefig(filename,dpi=fig.dpi, bbox_inches='tight')


def scenario_r_selection(problem_gen, num_problems, algorithms, r_s):

    estimates = {}
    converged = {}

    for rp in range(num_problems):
        print("+++++++++++++++++++++ Problem no.", rp+1, "+++++++++++++++++++++")
        start = cpu_time()
        f = problem_gen()
        f.toExp()
        end = cpu_time()
        print("[time] Model generation:", end - start)

        start = cpu_time()
        aux = np.log(np.sum(f.joint().t))
        end = cpu_time()
        print("[time] Z real value:", end - start)

        for r_alg in algorithms:
            for i_r, r in enumerate(r_s):
                alg = r_alg(r)
                if alg.name not in estimates:
                    estimates[alg.name] = np.zeros((num_problems, len(r_s)))
                    converged[alg.name] = np.zeros((num_problems, len(r_s)))

                print("Testing algorithm:",alg.name,"with r=",r)
                start = cpu_time()
                alg.set_problem(f.copy()) # alg.project_onto_q() set_problem now includes projection
                end = cpu_time()
                print("[time] Projection:", end - start)

                #def is_converged(self):
                #    return self.convergence

                estimates[alg.name][rp, i_r] = alg.get_convergence_distance()
                print("dist",estimates[alg.name][rp, i_r])
                converged[alg.name][rp, i_r] = alg.is_projection_converged()

    return estimates, converged

def plot_results_r_selection(r_s, estimates, converged, alg_reference="VIS-R-u", logscale=None, agg_func=np.median):

    i_r2 = np.where(r_s == 2.0)[0]

    aux = estimates[alg_reference].copy()
    ref_mean_per_rp = aux[:, i_r2]
    aux -= ref_mean_per_rp

    ref_max_per_rp = np.max(aux, axis=1, keepdims=True)
    #aux /= ref_max_per_rp


    colors = ["C" + s for s in list(map(str, np.arange(len(estimates))+1))]
    xaxis = np.arange(0, len(r_s)+1)[1:]

    fig, ax = plt.subplots(1, 3, figsize=(24, 8)) # 10,3

    for col, alg_name in enumerate(estimates):
        # print(estimates[alg_name].shape)

        '''ax[0].plot(xaxis, np.mean(estimates[alg_name], axis=(0, 1)), label=alg_name, color=colors[col])
        ax[0].set_title("Estimation")
        ax[0].set_xlabel("Num. samples")
        ax[0].set_ylabel("Estimation")
        ax[0].legend()'''
        point_rel_error = (estimates[alg_name] - ref_mean_per_rp)/ref_max_per_rp
        median_error = agg_func(point_rel_error, axis=0)
        ax[0].plot(xaxis, median_error, label=alg_name, color=colors[col])

        #high_pct_error = np.percentile(point_rel_error, 75, axis=(0, 1))
        #low_pct_error = np.percentile(point_rel_error, 25, axis=(0, 1))
        #ax[0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)

    ax[0].set_title("Relative distances for projections with different r")
    ax[0].set_xlabel("Renyi projection with r=")
    ax[0].set_ylabel("Rel. renyi-2 distance")
    ax[0].set_xticks(xaxis)
    ax[0].set_xticklabels(np.round(r_s,1))
    ax[0].legend()

    if (logscale is not None):
        ax[0].set_yscale('log')
        ax[0].set_ylabel("Log-" + ax[0].get_ylabel())
    #plt.show()




    for col, alg_name in enumerate(estimates):
        aux = estimates[alg_name].copy()
        ref_mean_per_rp = np.min(aux, axis=1, keepdims=True)#aux[:, i_r2]
        print()
        aux -= ref_mean_per_rp
        ref_max_per_rp = np.max(aux, axis=1, keepdims=True)

        point_rel_error = (estimates[alg_name] - ref_mean_per_rp)/ref_max_per_rp
        median_error = agg_func(point_rel_error, axis=0)
        print(median_error)
        ax[1].plot(xaxis, median_error, label=alg_name, color=colors[col])

        #high_pct_error = np.percentile(point_rel_error, 75, axis=(0, 1))
        #low_pct_error = np.percentile(point_rel_error, 25, axis=(0, 1))
        #ax[0].fill_between(xaxis, low_pct_error, high_pct_error, color=colors[col], alpha=0.2)

    ax[1].set_title("Relative distances for projections with different r")
    ax[1].set_xlabel("Renyi projection with r=")
    ax[1].set_ylabel("Rel. renyi-2 distance")
    ax[1].set_xticks(xaxis)
    ax[1].set_xticklabels(np.round(r_s,1))
    ax[1].legend()


    # diagrama de barras

    total_width = 0.9  # the width of the bars
    width = total_width / len(estimates)
    rel_points = np.linspace(-total_width/2, total_width/2, len(estimates)+1)

    for col, alg_name in enumerate(estimates):
        ax[2].bar(xaxis + rel_points[col], np.sum(converged[alg_name],axis=0), width, color=colors[col], label=alg_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[2].set_ylabel('Num. convergences')
    ax[2].set_title('Convergences by projection algorithm')
    ax[2].set_xticks(xaxis)
    ax[2].set_xticklabels(np.round(r_s, 1))
    ax[2].set_xlabel("Renyi projection with r=")
    #ax[2].legend()


    fig.tight_layout()

    plt.show()


def individual_plot_results_r_selection(r_s, estimates, converged, alg_reference="VIS-R-u", logscale=None):

    i_r2 = np.where(r_s == 2.0)[0]

    aux = estimates[alg_reference].copy()
    ref_mean_per_rp = aux[:, i_r2]
    aux -= ref_mean_per_rp

    ref_max_per_rp = np.max(aux, axis=1, keepdims=True)
    #aux /= ref_max_per_rp


    colors = ["C" + s for s in list(map(str, np.arange(len(estimates))+1))]
    xaxis = np.arange(0, len(r_s)+1)[1:]

    fig, ax = plt.subplots(10, 3, figsize=(24, 50)) # 10,3

    for col, alg_name in enumerate(estimates):
        # print(estimates[alg_name].shape)

        '''ax[0].plot(xaxis, np.mean(estimates[alg_name], axis=(0, 1)), label=alg_name, color=colors[col])
        ax[0].set_title("Estimation")
        ax[0].set_xlabel("Num. samples")
        ax[0].set_ylabel("Estimation")
        ax[0].legend()'''
        point_rel_error = (estimates[alg_name] - ref_mean_per_rp)/ref_max_per_rp
        for rep in np.arange(len(ref_mean_per_rp)):
            ax[rep,0].plot(xaxis, point_rel_error[rep,:], label=alg_name, color=colors[col])


    for rep in np.arange(len(ref_mean_per_rp)):
        ax[rep,0].set_title("Relative distances for projections with different r")
        ax[rep,0].set_xlabel("Renyi projection with r=")
        ax[rep,0].set_ylabel("Rel. renyi-2 distance")
        ax[rep,0].set_xticks(xaxis)
        ax[rep,0].set_xticklabels(np.round(r_s, 1))
        ax[rep,0].legend()
        if (logscale is not None):
            ax[rep,0].set_yscale('log')
            ax[rep,0].set_ylabel("Log-" + ax[rep,0].get_ylabel())


    for col, alg_name in enumerate(estimates):
        aux = estimates[alg_name].copy()
        ref_mean_per_rp = np.min(aux, axis=1, keepdims=True)#aux[:, i_r2]
        print()
        aux -= ref_mean_per_rp
        ref_max_per_rp = np.max(aux, axis=1, keepdims=True)

        point_rel_error = (estimates[alg_name] - ref_mean_per_rp)/ref_max_per_rp
        for rep in np.arange(len(ref_mean_per_rp)):
            ax[rep,1].plot(xaxis, point_rel_error[rep,:], label=alg_name, color=colors[col])

    for rep in np.arange(len(ref_mean_per_rp)):
        #ax[rep,1].set_title("Relative distances for projections with different r")
        ax[rep,1].set_xlabel("Renyi projection with r=")
        #ax[rep,1].set_ylabel("Rel. renyi-2 distance")
        ax[rep,1].set_xticks(xaxis)
        ax[rep,1].set_xticklabels(np.round(r_s,1))
        #ax[1].legend()


    # diagrama de barras

    total_width = 0.9  # the width of the bars
    width = total_width / len(estimates)
    rel_points = np.linspace(-total_width/2, total_width/2, len(estimates)+1)

    for col, alg_name in enumerate(estimates):
        for rep in np.arange(len(ref_mean_per_rp)):
            ax[rep,2].bar(xaxis + rel_points[col], converged[alg_name][rep,:], width, color=colors[col], label=alg_name)

    for rep in np.arange(len(ref_mean_per_rp)):
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[rep,2].set_ylabel('Num. convergences')
        ax[rep,2].set_title('Convergences by projection algorithm')
        ax[rep,2].set_xticks(xaxis)
        ax[rep,2].set_xticklabels(np.round(r_s, 1))
        ax[rep,2].set_xlabel("Renyi projection with r=")
        #ax[2].legend()


    fig.tight_layout()

    plt.show()