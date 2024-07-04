import copy
from utils.CPG_network import rand_CPG_network, CPG_network
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.utils import search_file_list
from scipy import stats
import itertools


def fitness_trial(trial_time_series, target_time_serie, window_size, fitness):
    best_ind = 0
    fitnesses = []
    for ii in range(trial_time_series.shape[1] - window_size):
        time_serie = trial_time_series[:, ii:ii + window_size + 1]
        fitness_t = np.mean(np.abs(time_serie - target_time_serie))
        if fitness_t < fitness:
            best_ind = ii
            fitness = fitness_t
        fitnesses.append(fitness)
    genome = trial_time_series[:, best_ind].squeeze()
    return fitness, genome, fitnesses

def generate_data(results_dir, targets, n_runs):
    dt = 0.15
    n_windows = [50, 100, 500, 1000]
    trial_ratios = [1, 1.25, 2, 5]
    n_samples = [100, 250, 500, 1000]

    for trial_ratio in trial_ratios:
        for n_sample in n_samples:
            for n_window in n_windows:
                n_trial_samples = int(trial_ratio*n_window)
                n_trials = n_sample

                controller_data = targets[f'{trial_ratio}'][f'{n_sample}']
                target_network = CPG_network(np.array([0] * n_cpgs), dt)
                target_network.set_A(controller_data[0])
                target_network.set_bias(controller_data[1])
                target_network.reset_controller()
                for _ in range(n_window):
                    target_network.update_CPG()
                target_time_series = target_network.y[1::2]

                for ind in range(n_runs):
                    n_network = len(target_network.A)

                    learner_res_dir = f'{results_dir}/ratio_{trial_ratio}/{n_sample}/w_{n_window}/{ind}'
                    if not os.path.exists(learner_res_dir):
                        os.makedirs(learner_res_dir, exist_ok=True)

                    if os.path.exists(learner_res_dir + '/' + 'x_best.npy'):
                        print(f"Completed experiment hyper_params ISO: {learner_res_dir}->{ind}")
                        continue
                    print(f"Starting experiment hyper_params ISO: {learner_res_dir}->{ind}")

                    weights = np.random.uniform(-1, 1, target_network.n_weights)
                    init_A = np.zeros_like(target_network.A)
                    init_A[target_network.weight_map] = weights
                    init_A -= init_A.T
                    np.save(f'{learner_res_dir}/weights.npy', init_A)

                    all_evals = []
                    fitnesses = []
                    genomes = []
                    f_best_so_far = []
                    x_best_so_far = []
                    prev_best = np.inf
                    x0 = np.random.uniform(-1, 1, (n_trials, n_network))

                    for x in x0:
                        controller = copy.deepcopy(target_network)
                        controller.set_A(init_A)
                        controller.set_bias(x)
                        controller.reset_controller()
                        for _ in range(n_trial_samples):
                            controller.update_CPG()
                        trial_time_series = controller.y[1::2]

                        trial_fitness, trial_genome, trial_fitnesses = fitness_trial(trial_time_series, target_time_series, n_window, prev_best)

                        if trial_fitness < prev_best:
                            f_best_so_far.append(trial_fitness)
                            x_best_so_far.append(trial_genome)
                            prev_best = trial_fitness
                        else:
                            f_best_so_far.append(f_best_so_far[-1])
                            x_best_so_far.append(x_best_so_far[-1])
                        all_evals += trial_fitnesses
                        fitnesses.append(trial_fitness)
                        genomes.append(trial_genome)

                    # log results
                    np.save(learner_res_dir + '/' + 'all_evals', np.array(all_evals))
                    np.save(learner_res_dir + '/' + 'fitnesses', np.array(fitnesses))
                    np.save(learner_res_dir + '/' + 'genomes', np.array(genomes))
                    np.save(learner_res_dir + '/' + 'f_best', np.array(f_best_so_far))
                    np.save(learner_res_dir + '/' + 'x_best', np.array(x_best_so_far))
                    print(f"Finished experiment hyper_params ISO: {learner_res_dir}->{ind}")


if __name__ == "__main__":
    # %% Generate data
    n_cpgs = 10
    n_runs = 30
    n_windows = [50, 100, 500, 1000]
    trial_ratios = [1, 1.25, 2, 5]
    n_samples = [100, 250, 500, 1000]
    results_dir = './results/APPENDIX/hyper_params'

    if not os.path.exists('./exp_APPENDIX/hyper_params.npy') and True:
        targets = {}
        for trial_ratio in trial_ratios:
            targets[f'{trial_ratio}'] = {}
            for n_sample in n_samples:
                network = rand_CPG_network(n_cpgs, 0.3)
                initial_states = np.random.uniform(-1, 1, (n_cpgs*2,))
                targets[f'{trial_ratio}'][f'{n_sample}'] = [network, initial_states]
        np.save('./exp_APPENDIX/hyper_params', targets)
    target_data = np.load('./exp_APPENDIX/hyper_params.npy', allow_pickle=True)
    target_data = dict(target_data.item())

    generate_data(results_dir, target_data, n_runs)
    if not os.path.exists(f'{results_dir}/ratio_{trial_ratios[-1]}/{n_samples[-1]}/w_{n_windows[-1]}/{n_runs - 1}'):
        generate_data(results_dir, target_data, n_runs)
    else:
        print("ISO DATA already generated, continue analysis")

    # %% Data Analysis
    DATA = []
    final_pop_means = []
    for trial_ratio, n_sample in itertools.product(trial_ratios, n_samples):
        figure, ax = plt.subplots(figsize=(5, 4))
        for n_window in n_windows:
            n_trial_samples = int(trial_ratio * n_window)
            n_trials = int(n_sample / n_trial_samples)

            exp_name = f"ratio_{trial_ratio}/{n_sample}"
            filenames_f = search_file_list(f'./{results_dir}/{exp_name}/w_{n_window}', 'f_best.npy')
            combined_data = np.array([np.load(fname) for fname in filenames_f])
            f_mean = combined_data.mean(axis=0)
            SE95 = f_mean.std(axis=0) / np.sqrt(len(filenames_f)) * 1.96
            f_max = combined_data.max()
            f_min = combined_data.min()
            DATA.append((combined_data, SE95, f_max, f_min))
            final_pop_means.append(combined_data[:, -1])


            x_axis = np.arange(len(f_mean)) + 1
            ax.plot(x_axis, f_mean, label=r"$W_{{{}}}$".format(n_window))
            ax.fill_between(x_axis, f_mean - SE95, f_mean + SE95, alpha=.5)
            ax.set_ylim([0.35, 0.7])

        # ax.set_xlabel('Trials', size=16)
        if n_sample == 100:
            ax.set_ylabel('Mean absolute error', size=16)
        ax.grid()
        ax.legend()
        figure.tight_layout()
        figure.savefig(f"{results_dir}/APP_r{trial_ratio}:{n_sample}_curve.pdf", bbox_inches='tight')
        plt.close()

        F, p = stats.f_oneway(*final_pop_means)
        print(F, p)
    print("FINISHED")
