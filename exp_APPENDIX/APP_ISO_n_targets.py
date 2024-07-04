import copy
from utils.CPG_network import rand_CPG_network, CPG_network
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.utils import search_file_list
from scipy import stats


def fitness_target(trial_time_series, target_time_serie):
    fitness = np.inf
    best_ind = 0
    for ii in range(trial_time_series.shape[1] - 400):
        time_serie = trial_time_series[:, ii:ii + 401]
        fitness_t = np.mean(np.abs(time_serie - target_time_serie))
        if fitness_t < fitness:
            best_ind = ii
            fitness = fitness_t
    genome = trial_time_series[:, best_ind].squeeze()
    return fitness, genome

def generate_data(results_dir, targets, n_runs):
    dt = 0.15
    t_window = np.arange(0, 60, dt)
    t_trial = np.arange(0, 120, dt)
    n_trials = 150
    parse_trial = np.vectorize(fitness_target, signature=f'({n_cpgs},n),({n_cpgs},m)->(), (k)')

    for n_target in targets:
        target_time_series = []
        controller_data = targets[n_target]
        for i_target in range(int(n_target)):
            target_network = CPG_network(np.array([0] * n_cpgs), dt)
            target_network.set_A(controller_data[0][i_target])
            target_network.set_bias(controller_data[1][i_target])
            target_network.reset_controller()
            for _ in t_window:
                target_network.update_CPG()
            target_time_series.append(target_network.y[1::2])

        for ind in range(n_runs):
            n_network = len(target_network.A)

            learner_res_dir = f'{results_dir}/{n_target}_targets/{ind}'
            if not os.path.exists(learner_res_dir):
                os.makedirs(learner_res_dir, exist_ok=True)

            if os.path.exists(learner_res_dir + '/' + 'x_best.npy'):
                print(f"Completed experiment n_targets ISO: {n_target}->{ind}")
                continue
            print(f"Starting experiment n_targets ISO: {n_target}->{ind}")

            weights = np.random.uniform(-1, 1, target_network.n_weights)
            init_A = np.zeros_like(target_network.A)
            init_A[target_network.weight_map] = weights
            init_A -= init_A.T
            np.save(f'{learner_res_dir}/weights.npy', init_A)

            fitnesses = []
            genomes = []
            f_best_so_far = []
            x_best_so_far = []
            prev_best = [np.inf]*int(n_target)
            x0 = np.random.uniform(-1, 1, (n_trials, n_network))
            for x in x0:
                controller = copy.deepcopy(target_network)
                controller.set_A(init_A)
                controller.set_bias(x)
                controller.reset_controller()
                for _ in t_trial:
                    controller.update_CPG()
                trial_time_series = controller.y[1::2]

                f_best_so_far_trial = []
                x_best_so_far_trial = []
                trial_fitness, trial_genome = parse_trial(trial_time_series, target_time_series)

                for i_target in range(int(n_target)):
                    if trial_fitness[i_target] < prev_best[i_target]:
                        f_best_so_far_trial.append(trial_fitness[i_target])
                        x_best_so_far_trial.append(trial_genome[i_target])
                        prev_best[i_target] = trial_fitness[i_target]
                    else:
                        f_best_so_far_trial.append(f_best_so_far[-1][i_target])
                        x_best_so_far_trial.append(x_best_so_far[-1][i_target])
                fitnesses.append(trial_fitness)
                genomes.append(trial_genome)
                f_best_so_far.append(f_best_so_far_trial)
                x_best_so_far.append(x_best_so_far_trial)

            # log results
            np.save(learner_res_dir + '/' + 'fitnesses', np.array(fitnesses))
            np.save(learner_res_dir + '/' + 'genomes', np.array(genomes))
            np.save(learner_res_dir + '/' + 'f_best', np.array(f_best_so_far))
            np.save(learner_res_dir + '/' + 'x_best', np.array(x_best_so_far))
            print(f"Finished experiment ISO: n_targets ISO: {n_target}->{ind} {f_best_so_far[-1]}")


if __name__ == "__main__":
    # %% Generate data
    n_cpgs = 10
    n_runs = 30
    n_targets_list = [3, 5, 50, 100]
    results_dir = './results/APPENDIX/n_targets/'

    if not os.path.exists('./exp_APPENDIX/n_targets.npy') and True:
        targets = {}
        for n_targets in n_targets_list:
            n_networks = [rand_CPG_network(n_cpgs, 0.3) for _ in range(n_targets)]
            n_initial_states = [np.random.uniform(-1, 1, (n_cpgs*2,)) for _ in range(n_targets)]
            targets[str(n_targets)] = [n_networks, n_initial_states]
        np.save('./exp_APPENDIX/n_targets', targets)
    target_data = np.load('./exp_APPENDIX/n_targets.npy', allow_pickle=True)
    target_data = dict(target_data.item())

    if not os.path.exists(f'{results_dir}/{n_targets_list[-1]}_targets/{n_runs - 1}'):
        generate_data(results_dir, target_data, n_runs)
    else:
        print("ISO DATA already generated, continue analysis")

    # %% Data Analysis
    exp_name = [f"{n_target}_targets" for n_target in n_targets_list]
    DATA = []
    final_pop_means = []
    figure, ax = plt.subplots(figsize=(5, 4))
    for exp_n, experiment in zip(n_targets_list, exp_name):
        filenames_f = search_file_list(f'./{results_dir}/{experiment}', 'f_best.npy')
        filenames_x_best = search_file_list(f'./{results_dir}/{experiment}', 'x_best.npy')
        filenames_x_init = search_file_list(f'./{results_dir}/{experiment}', 'genomes.npy')
        combined_data = np.array([np.load(fname) for fname in filenames_f])
        x_best_data = np.array([np.load(fname) for fname in filenames_x_best])
        targets_mean = combined_data.mean(axis=2)
        targets_min = combined_data.min(axis=2)
        targets_max = combined_data.max(axis=2)
        f_mean = targets_mean.mean(axis=0)
        f_min_ = targets_min.mean(axis=0)
        f_max_ = targets_max.mean(axis=0)
        SE95 = f_mean.std(axis=0) / np.sqrt(len(filenames_f)) * 1.96
        f_max = combined_data.max()
        f_min = combined_data.min()
        DATA.append((combined_data, SE95, f_max, f_min))
        final_pop_means.append(targets_mean[:, -1])

        ax.plot(np.arange(0, len(f_mean))*2, f_mean, label=r'$T_{{{}}}$'.format(exp_n))
        ax.fill_between(np.arange(0, len(f_mean))*2, f_mean - SE95, f_mean + SE95, alpha=.5)

    ax.set_xlabel('time [min]', size=16)
    ax.set_ylabel('Mean absolute error', size=16)
    ax.grid()
    ax.legend()
    figure.tight_layout()
    figure.savefig(f"{results_dir}/APP_n_curve.pdf", bbox_inches='tight')

    F, p = stats.f_oneway(*final_pop_means)
    print(F, p)
    print("FINISHED")
