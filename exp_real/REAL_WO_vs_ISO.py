import os
import sys

sys.path.extend([os.getcwd()])
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from ISO import configuration_file
from utils.utils import robot_names


def generate_data_iso(results_dir):
    params = configuration_file
    reps = params['n_reps']
    n_trials = params['n_trials']
    trial_time = params['trial_time']
    for robot in robot_names:
        os.system(f'{os.path.join("exp_real", "ISO.py")} --robot {robot}')
        for rep in range(reps):
            res_dir = os.path.join(results_dir, 'ISO', robot, f'{robot}_n{n_trials}_{trial_time}.{rep}')
            os.system(f'python {os.path.join("exp_real", "retest_CPG.py")} --dir "{res_dir}"')


def generate_data_wo(results_dir):
    for robot in robot_names:
        res_dir = os.path.join(results_dir, 'WO', robot)
        os.makedirs(res_dir)
        for root, dirs, files in os.walk(res_dir):
            for file in files:
                print(os.path.join(root, file))
                shutil.copy(os.path.join(root, file), res_dir)
        os.system(f'python retest_CPG.py --dir "{res_dir}"')


if __name__ == "__main__":
    names = ['ant', 'gecko', 'spider', 'blokky', 'salamander', 'stingray']
    skills = ['gait', 'left', 'right']
    skill_title = ['→', '⟲', '⟳']
    skill_vel_norm = [0.6, 60, 60]

    # %% Generate data
    results_dir = os.path.join('results', 'REAL')
    if not os.path.exists(f'{results_dir}/WO/stingray/stingray_right/fitnesses_trial.npy'):
        generate_data_wo(results_dir)
    if not os.path.exists(f'{results_dir}/ISO/stingray/stingray_n5_120.2/stingray_right/fitnesses_trial.npy'):
        generate_data_iso(results_dir)

    # %% Data Analysis
    max_dict_real = {}
    for name in names:
        wo_performance = []
        print(name + ' real wo results')
        for ii, skill in enumerate(skills):
            wo_skill = np.load(f'{results_dir}/WO/{name}/{name}_{skill}/fitnesses_trial.npy', allow_pickle=True)
            wo_performance.append(np.abs(wo_skill)[ii]/skill_vel_norm[ii])
        max_dict_real[name] = wo_performance

    exp_name = ['ISO', 'WO']
    for name in names:
        figure, ax = plt.subplots(3, 1, figsize=(4, 5))
        ranks = []
        f = np.empty((3, 3))
        for ii, skill in enumerate(skills):
            for repetition in range(3):
                exp_folder = os.path.join(f'{results_dir}/ISO/'
                                          f'{name}/{name}_n5_120.{repetition}/{name}_{skill}/fitnesses_trial.npy')
                f[repetition, ii] = np.load(exp_folder, allow_pickle=True)[ii]

            sort_ind = np.argsort(f[:, ii], axis=0)
            rank = np.empty_like(sort_ind)
            rank[sort_ind] = np.arange(len(sort_ind))
            ranks.append(rank)

        f /= np.array([skill_vel_norm, skill_vel_norm, skill_vel_norm])
        sum_rank = np.sum(ranks, axis=0)
        norm_bars, best_bar = np.delete(f, sum_rank.argmax(), axis=0), f[sum_rank.argmax(), :]
        print(name, sum_rank.argmax())
        f_ranked = np.vstack((norm_bars, best_bar))
        f = f_ranked
        for ii, skill in enumerate(skills):
            rob_max_real = max_dict_real[name][ii]
            ax[ii].bar(np.arange(0, 3), f[:, ii], label='real')
            ax[ii].bar(2, f[-1:, ii], color='None', edgecolor='k', label='real', hatch="//")
            ax[ii].bar(3.5, rob_max_real, label='transfer')

            ax[ii].xaxis.label.set_size(15)
            ax[ii].yaxis.label.set_size(15)
            ax[ii].set_title(skill_title[ii], fontsize=20)

            ax[ii].set_xticks([0, 1, 2, 3.5,], ['ISO 1', 'ISO 2', 'ISO 3', 'WO* trans'])
            ax[ii].set_ylabel('cm/s' if skill == 'gait' else 'rad/s')

            print(f[:, ii].mean().__round__(3), rob_max_real.__round__(3))
        figure.tight_layout()
        figure.savefig(f"{name}_real.pdf")
        # figure.show()
        plt.close(figure)
