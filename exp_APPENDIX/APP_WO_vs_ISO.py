import copy

from scipy import stats

from utils.CPG_network import rand_CPG_network, CPG_network
from utils.Learners import DifferentialEvolution
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.utils import search_file_list


def fitness_wo(x, target_controller):
    controller = copy.deepcopy(target_controller)
    controller.set_weights(x)
    controller.reset_controller()
    for _ in np.arange(0, 60, 0.05):
        controller.update_CPG()
    return np.sum(np.absolute(controller.y - target_controller.y)) / target_controller.y.size


def fitness_iso(x, target_controller):
    controller = copy.deepcopy(target_controller)
    controller.set_A(x[1])
    controller.set_bias(x[0])
    controller.reset_controller()
    for _ in np.arange(0, 60, 0.05):
        controller.update_CPG()
    return np.sum(np.absolute(controller.y - target_controller.y)) / target_controller.y.size

def generate_data_wo(results_dir, target_network, n_runs):
    for _ in np.arange(0, 60, 0.05):
        target_network.update_CPG()

    params = {}
    pop_size = 30
    n_generations = 100
    params['evaluate_objective_type'] = 'full'
    params['pop_size'] = pop_size
    params['CR'] = 0.9
    params['F'] = 0.3

    f_best = []
    x_best = []
    for ind in range(n_runs):
        print(f"Starting experiment WO: {ind}")
        x0 = np.random.uniform(-1, 1, (pop_size, target_network.n_weights))
        learner_res_dir = f'{results_dir}/WO/{ind}'
        if os.path.exists(learner_res_dir + '/' + 'x_best.npy'):
            print(f"Completed experiment n_skills WO: {ind}")
            continue
        if not os.path.exists(learner_res_dir):
            os.makedirs(learner_res_dir, exist_ok=True)
        learner = DifferentialEvolution(x0, 1, 'revde', (-1, 1), params, output_dir=learner_res_dir)
        for gen in range(n_generations):
            fitnesses = []
            for x in learner.x_new:
                fitness = fitness_wo(x, target_network)
                fitnesses.append(fitness)
            learner.f = np.array(fitnesses)
            learner.x = learner.x_new
            learner.x_new = np.array([])
            _ = learner.get_new_genomes()
        f_best.append(learner.f_best_so_far)
        x_best.append(learner.x_best_so_far)

        # log results
        learner.save_results()
        print(f"Finished experiment WO: {ind} | Best score: {learner.f_best_so_far[-1]}")


def generate_data_iso(results_dir, target_network, n_runs):
    t = np.arange(0, 60, 0.05)
    for _ in t:
        target_network.update_CPG()

    params = {}
    pop_size = 30
    n_generations = 100
    params['evaluate_objective_type'] = 'full'
    params['pop_size'] = pop_size
    params['CR'] = 0.9
    params['F'] = 0.3

    f_best = []
    x_best = []
    n_network = len(target_network.A)  # we optimize initial states instead of weights
    for ind in range(n_runs):
        print(f"Starting experiment ISO: {ind}")
        x0 = np.random.uniform(-1, 1, (pop_size, n_network))
        learner_res_dir = f'{results_dir}/ISO/{ind}'
        if os.path.exists(learner_res_dir + '/' + 'x_best.npy'):
            print(f"Completed experiment ISO: {ind}")
            continue
        if not os.path.exists(learner_res_dir):
            os.makedirs(learner_res_dir, exist_ok=True)
        learner = DifferentialEvolution(x0, 1, 'revde', (-1, 1), params, output_dir=learner_res_dir)

        # Create random CPG reservoir
        weights = np.random.uniform(-1, 1, target_network.n_weights)
        init_A = np.zeros_like(target_network.A)
        init_A[target_network.weight_map] = weights
        init_A -= init_A.T
        reservoir_A = init_A
        np.save(f'{learner_res_dir}/weights.npy', init_A)

        for gen in range(n_generations):
            fitnesses = []
            for x in learner.x_new:
                fitness = fitness_iso((x, reservoir_A), target_network)
                fitnesses.append(fitness)
            learner.f = np.array(fitnesses)
            learner.x = learner.x_new
            learner.x_new = np.array([])
            _ = learner.get_new_genomes()

        f_best.append(learner.f_best_so_far)
        x_best.append(learner.x_best_so_far)

        # log results
        learner.save_results()
        print(f"Finished experiment ISO: {ind} | Best score: {learner.f_best_so_far[-1]}")


if __name__ == "__main__":
    # %% Generate data
    results_dir = './results/APPENDIX/WO_vs_ISO'
    n_cpgs = 8
    n_runs = 30
    if not os.path.exists('./exp_APPENDIX/weights_wo_vs_iso.npy'):
        target_matrix = rand_CPG_network(n_cpgs, 0.3)
        np.save('./exp_APPENDIX/weights_wo_vs_iso.npy', target_matrix)

    target_matrix = np.load('./exp_APPENDIX/weights_wo_vs_iso.npy')
    target_weights = target_matrix[np.nonzero(np.triu(target_matrix))]

    target_A = target_matrix
    target_network = CPG_network(np.array([0] * n_cpgs), 0.05)
    target_network.set_A(target_A)
    target_network.reset_controller()
    if not os.path.exists(f'{results_dir}/WO/{n_runs - 1}'):
        generate_data_wo(results_dir, target_network, n_runs)
    else:
        print("WO DATA already generated, continue ISO experiment")
    if not os.path.exists(f'{results_dir}/ISO/{n_runs - 1}'):
        generate_data_iso(results_dir, target_network, n_runs)
    else:
        print("ISO DATA already generated, continue analysis")

    # %% Data Analysis
    exp_name = ['ISO', 'WO']
    DATA = []
    final_pop_means = []
    figure, ax = plt.subplots(figsize=(4, 3))
    for experiment in exp_name:
        filenames_f = search_file_list(f'./{results_dir}/{experiment}', 'f_best.npy')
        filenames_x_best = search_file_list(f'./{results_dir}/{experiment}', 'x_best.npy')
        filenames_x_init = search_file_list(f'./{results_dir}/{experiment}', 'genomes.npy')
        combined_data = np.array([np.load(fname) for fname in filenames_f])
        x_best_data = np.array([np.load(fname) for fname in filenames_x_best])
        f_mean = combined_data.mean(axis=0)
        SE95 = combined_data.std(axis=0) / np.sqrt(len(filenames_f)) * 1.96
        f_max = combined_data.max()
        f_min = combined_data.min()
        DATA.append((combined_data, SE95, f_max, f_min))
        final_pop_means.append(combined_data[:, -1])

        ax.plot(f_mean, label=experiment)
        ax.fill_between(np.arange(0, len(f_mean)), f_mean - SE95, f_mean + SE95, alpha=.5)

    ax.set_xlabel('Generations', size=10)
    ax.set_ylabel('Mean absolute error', size=10)
    ax.grid()
    ax.legend()
    figure.tight_layout()
    figure.savefig(f"{results_dir}/APP_curve.pdf", bbox_inches='tight')
    F, p = stats.ttest_ind(*final_pop_means)
    print(F, p)
    print("FINISHED")

