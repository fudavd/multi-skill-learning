import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.CPG_network import rand_CPG_network, CPG_network
from utils.Learners import DifferentialEvolution


def fitness_cpg(x, target_controller):
    controller = copy.deepcopy(target_controller)
    controller.set_weights(x)
    controller.reset_controller()
    for _ in np.arange(0, 60, 0.05):
        controller.update_CPG()
    return np.sum(np.absolute(controller.y - target_controller.y)) / target_controller.y.size


def generate_data(results_dir, target_network, n_runs):
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
        print(f"Starting experiment: {ind}")
        x0 = np.random.uniform(-1, 1, (pop_size, target_network.n_weights))
        learner_res_dir = f'{results_dir}/learner/{ind}'
        if not os.path.exists(learner_res_dir):
            os.makedirs(learner_res_dir, exist_ok=True)
        learner = DifferentialEvolution(x0, 1, 'revde', (-1, 1), params, output_dir=learner_res_dir)
        for gen in range(n_generations):
            fitnesses = []
            for x in learner.x_new:
                fitness = fitness_cpg(x, target_network)
                fitnesses.append(fitness)
            learner.f = np.array(fitnesses)
            learner.x = learner.x_new
            learner.x_new = np.array([])
            _ = learner.get_new_genomes()
        f_best.append(learner.f_best_so_far)
        x_best.append(learner.x_best_so_far)
        # log results
        learner.save_results()
        print(f"Finished experiment: {ind} | Best score: {learner.f_best_so_far[-1]}")


if __name__ == "__main__":
    # %% Generate data
    results_dir = './results/APPENDIX/optimising_weights'
    n_cpgs = 8
    n_runs = 30
    if not os.path.exists('./exp_APPENDIX/weights_optimising.npy'):
        target_matrix = rand_CPG_network(n_cpgs, 0.3)
        np.save('./exp_APPENDIX/weights_optimising.npy', target_matrix)

    target_matrix = np.load('./exp_APPENDIX/weights_optimising.npy')
    target_weights = target_matrix[np.nonzero(np.triu(target_matrix))]

    target_A = target_matrix
    target_network = CPG_network(np.array([0] * n_cpgs), 0.05)
    target_network.set_A(target_A)
    target_network.reset_controller()
    if not os.path.exists(f'{results_dir}/learner/{n_runs - 1}'):
        generate_data(results_dir, target_network, n_runs)
    else:
        print("DATA already generated, continue analysis")

    # %% Data Analysis
    weights_best = []
    dist_initial = []
    dist_best = []
    for run in range(n_runs):  # Calculate the eigenvalue distributions at the end of WO (best_CPG) and start init_CPG
        learner_res_dir = f'{results_dir}/learner/{run}'
        weights_learner = np.load(f'{learner_res_dir}/x_best.npy', allow_pickle=True)
        weights_best.append(weights_learner[-1].flatten())
        best_CPG = copy.deepcopy(target_network)
        best_CPG.set_weights(weights_best[-1])
        eig_val_best = np.linalg.eig(best_CPG.A)
        dist_best.append(eig_val_best[0].imag.tolist())

        genomes = np.load(f'{learner_res_dir}/genomes.npy')
        dist_init = []
        for i in range(genomes.shape[1]):
            weights_0 = genomes[0, i, :]
            init_CPG = copy.deepcopy(target_network)
            init_CPG.set_weights(weights_0)
            eig_val_init = np.linalg.eig(init_CPG.A)
            dist_init.append(eig_val_init[0].imag.tolist())
        dist_initial.append(dist_init)

    eig_val_target = np.linalg.eig(target_A)
    dist_target = np.array(eig_val_target[0].imag)
    dists = ["initial", "best", "target"]
    for ind, a in enumerate([dist_target, dist_initial, dist_best]):  # save eigenvalue distribution data
        eig_vec = np.array(a).reshape(-1, 1)
        np.save(f'{results_dir}/eigen_dist/dist_{dists[ind]}', eig_vec)

    # %% Create figures
    for ind, dist in enumerate(dists):
        eig_vec = np.load(f'{results_dir}/eigen_dist/dist_{dist}.npy', allow_pickle=True)
        bins = np.arange(np.floor(min(eig_vec * 10)) / 10 - 0.025, np.ceil(max(eig_vec) * 10) / 10 + 0.025, 0.05)
        _ = plt.hist(eig_vec, density=True, bins=bins, label=f'{dist.capitalize()}', alpha=0.5,
                     zorder=int(ind - 1.5 ** 2))
    plt.xlabel('Eigenvalues (im)')
    plt.ylabel('Density')
    plt.legend()
    plt.xticks(np.arange(-3, 4), ['-3$i$', '-2$i$', '-1$i$', '0', '1$i$', '2$i$', '3$i$'], usetex=True)
    plt.savefig(f'{results_dir}/APP_WO_eigen.pdf')
    plt.show()

    figure, ax = plt.subplots()
    ax.bar(np.arange(len(target_weights)) * 1.5 - 0.25, target_weights, width=0.5, color='g', label='Target')
    ax.bar(np.arange(len(target_weights)) * 1.5 + 0.25, np.mean(weights_best, axis=0), width=0.5, color='#FF800E',
           label='Best',
           yerr=np.std(weights_best, axis=0) / np.sqrt(n_runs) * 1.96, ecolor='k', capsize=3)

    ax.set_xticks(np.arange(0, len(target_weights)) * 1.5)
    w_label = ['$w_{' + f'{i + 1}' + '}$' for i in range(len(target_weights))]
    ax.set_xticklabels(w_label, usetex=True)
    ax.yaxis.grid(True)
    figure.tight_layout()
    figure.legend(loc=(0.65, 0.15))
    figure.show()
    figure.savefig(f"{results_dir}/APP_weights.pdf", bbox_inches='tight')
    print("FINISHED")
