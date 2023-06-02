import matplotlib.pyplot as plt
import numpy as np
import os
from utils.utils import search_file_list


def generate_data():
    robots = ['ant', 'gecko', 'spider', 'blokky', 'salamander', 'stingray']
    for robot in robots:
        os.system(f'./exp_sim/WO.py --robot {robot}')
    for robot in robots:
        os.system(f'./exp_sim/ISO.py --robot {robot}')


if __name__ == "__main__":
    generate_data()

    # %% Generate data
    results_dir = './results/SIM/'

    # %% Data Analysis
    exp_name = ['ISO', 'WO']
    DATA = []
    figure, ax = plt.subplots()
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

        ax.plot(f_mean, label=experiment)
        ax.fill_between(np.arange(0, len(f_mean)), f_mean - SE95, f_mean + SE95, alpha=.5)

    ax.set_xlabel('Generations', size=16)
    ax.set_ylabel('Mean absolute error', size=16)
    # ax.set_ylim(0, max())
    ax.grid()
    ax.legend()
    figure.tight_layout()
    # figure.set_size_inches(15, 9)
    figure.savefig(f"{results_dir}/APP_curve.pdf", bbox_inches='tight')
