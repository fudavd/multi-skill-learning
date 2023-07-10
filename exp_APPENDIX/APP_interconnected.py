import os.path

import numpy as np
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
from utils.CPG_network import rand_CPG_network
import matplotlib.pyplot as plt

rng = default_rng()
rvs = stats.uniform(-1, 2).rvs


def generate_data():
    for density in ds:
        eig_vals = []
        for k in range(10_000):
            mat = rand_CPG_network(n_oscillators, density)
            eig_val = np.linalg.eig(mat)
            eig_vals.append(eig_val[0].imag.tolist())
        eig_vector = np.array(eig_vals).reshape(-1, 1)
        np.save(f'{results_dir}/dist_{density}', eig_vector)


if __name__ == '__main__':
    results_dir = './results/APPENDIX/interconnected'
    ds = [0, 0.3, 1.0]
    n_oscillators = 10
    if not os.path.exists(f'{results_dir}/dist_1.0'):
        generate_data()
    else:
        print("DATA already generated, continue analysis")

    for ind, density in enumerate(ds):
        eig_vec = np.load(f'{results_dir}/dist_{density}.npy')
        _ = plt.hist(eig_vec, density=True, bins='auto', label=f'd={density}', alpha=0.5, zorder=int(ind - 1.5 ** 2))
    plt.xlabel('Eigenvalues (im)')
    plt.ylabel('Density')
    plt.legend()
    plt.xticks(np.arange(-4, 5), ['-4$i$', '-3$i$', '-2$i$', '-1$i$', '0', '1$i$', '2$i$', '3$i$', '4$i$'], usetex=True)
    plt.savefig(f'{results_dir}/APP_freq_CPG.pdf')
    plt.show()
    print("FINISHED")