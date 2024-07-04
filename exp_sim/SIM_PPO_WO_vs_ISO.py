import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from utils.utils import search_file_list, robot_names
from cycler import cycler

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.prop_cycle'] = cycler(
    color=['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
           '#CFCFCF'])

def generate_data():
    for robot in robot_names:
        os.system(f'python {os.path.join("exp_sim", "PPO.py")} --robot {robot} --headless')
    for robot in robot_names:
        os.system(f'python {os.path.join("exp_sim", "WO.py")} --robot {robot} --headless')
    print("WO DATA generated, continue ISO experiment")
    for robot in robot_names:
        os.system(f'python {os.path.join("exp_sim", "ISO.py")} --robot {robot} --headless')


if __name__ == "__main__":
    results_dir = os.path.join('results', 'SIM/')
    # %% Generate data
    if not os.path.exists(f'{results_dir}/ISO/{robot_names[-1]}'):
        generate_data()
    else:
        print("DATA already generated, continue analysis")

    # %% Data Analysis
    DATA = []
    controller_update_time = 0.1
    eval_time = 180
    window_time = 60
    N_samples = int((eval_time - window_time) / controller_update_time)
    tteq = []
    tteq_pure = []
    for name in robot_names:
        skills = ['gait', 'rot_l', 'rot_r']
        skills_r = ['gait', 'left', 'right']
        skill_title = ['→', '⟲', '⟳']
        figure, ax = plt.subplots(3, 1, figsize=(4, 5))
        tteq_rob = []
        bar_we = []
        bar_s0 = []
        max_we = []
        max_s0 = []
        for ii, skill in enumerate(skills):
            folder_state0 = os.path.join(results_dir, 'ISO', name, skill)
            folder_weight = os.path.join(results_dir, 'WO', name, skill)
            folder_ppo = os.path.join(results_dir, 'PPO', name, skill)
            state0_list = search_file_list(folder_state0, 'fitnesses.npy')
            weight_list = search_file_list(folder_weight, 'fitnesses.npy')
            ppo_list = search_file_list(folder_ppo, 'eps_rewards.npy')

            fitness_max_s0 = []
            peakindex_s0 = []
            for fitness_ref in state0_list:
                fitness_temp = np.load(fitness_ref, allow_pickle=True)

                temp_max = -np.inf
                max_index = []
                fitness_max_temp = []
                peakindex_temp = []
                for ind in range(150):
                    samples = np.arange(N_samples * ind, N_samples * (ind + 1))
                    curr_max = np.max(fitness_temp[samples])
                    if temp_max < curr_max:
                        temp_max = curr_max
                        peakindex_temp.append(ind)
                    fitness_max_temp.append(temp_max)

                fitness_max_s0.append(fitness_max_temp)
                peakindex_s0.append(peakindex_temp)

            fitness_max_we = []
            peakindex_we = []
            for fitness_ref in weight_list:
                fitness_temp = np.load(fitness_ref, allow_pickle=True)

                temp_max = -np.inf
                max_index = []
                fitness_max_temp = []
                peakindex_temp = []
                for ind in range(300):
                    curr_max = fitness_temp[ind]
                    if temp_max < curr_max:
                        temp_max = curr_max
                        peakindex_temp.append(ind)
                    fitness_max_temp.append(temp_max)

                fitness_max_we.append(fitness_max_temp)
                peakindex_we.append(peakindex_temp)

            fitness_max_ppo = []
            peakindex_ppo = []
            for fitness_ref in ppo_list:
                fitness_temp = np.load(fitness_ref, allow_pickle=True)

                temp_max = -np.inf
                max_index = []
                fitness_max_temp = []
                peakindex_temp = []
                for ind in range(300):
                    curr_max = fitness_temp[ind]
                    if temp_max < curr_max:
                        temp_max = curr_max
                        peakindex_temp.append(ind)
                    fitness_max_temp.append(temp_max)

                fitness_max_ppo.append(fitness_max_temp)
                peakindex_ppo.append(peakindex_temp)

            fitness_mean_s0 = np.repeat(np.mean(fitness_max_s0, axis=0), 2)
            fitness_std_s0 = np.repeat(np.std(fitness_max_s0, axis=0), 2)
            fitness_mean_we = np.mean(fitness_max_we, axis=0)
            fitness_std_we = np.std(fitness_max_we, axis=0)
            fitness_mean_ppo = np.mean(fitness_max_ppo, axis=0)
            fitness_std_ppo = np.std(fitness_max_ppo, axis=0)

            fitness_amax_s0 = np.argmax(fitness_max_s0, axis=0)[-1]
            s0_best = state0_list[fitness_amax_s0]
            s0_best_genome = np.load(s0_best.replace('fitnesses.npy', "x_best.npy"), allow_pickle=True)
            s0_best_weights = np.load(s0_best.replace('fitnesses.npy', "weights.npy"), allow_pickle=True)

            fitness_amax_we = np.argmax(fitness_max_we, axis=0)[-1]
            we_best = weight_list[fitness_amax_we].replace('fitnesses.npy', "x_best.npy")
            we_best_genome = np.load(we_best.replace('fitnesses.npy', "x_best.npy"), allow_pickle=True)[-1]

            fitness_amax_ppo = np.argmax(fitness_max_ppo, axis=0)[-1]

            error_norm = np.sqrt(30)
            ax[ii].plot(np.arange(0, 300), fitness_mean_s0, label='state0')
            ax[ii].fill_between(np.arange(0, 300), fitness_mean_s0 - fitness_std_s0 / error_norm,
                                fitness_mean_s0 + fitness_std_s0 / error_norm,
                                alpha=.5)

            ax[ii].plot(np.arange(0, 300), fitness_mean_we, label='weights')
            ax[ii].fill_between(np.arange(0, 300), fitness_mean_we - fitness_std_we / error_norm,
                                fitness_mean_we + fitness_std_we / error_norm,
                                alpha=.5)

            ax[ii].plot(np.arange(0, 300), fitness_mean_ppo, label='PPO')
            ax[ii].fill_between(np.arange(0, 300), fitness_mean_ppo - fitness_std_ppo / error_norm,
                                fitness_mean_ppo + fitness_std_ppo / error_norm,
                                alpha=.5)
            max_we.append(fitness_mean_we.max())
            max_s0.append(fitness_mean_s0.max())

            mean_norm_we = fitness_mean_we / max_we[-1]
            mean_norm_s0 = fitness_mean_s0 / max_we[-1]
            we_perf = []
            s0_perf = []
            for ind in range(1, 5):
                try:
                    we_time = np.where(mean_norm_we >= 0.25 * ind)[0][0]
                except:
                    we_time = np.nan
                try:
                    s0_time = np.where(mean_norm_s0 >= 0.25 * ind)[0][0]
                except:
                    s0_time = np.nan
                we_perf.append(we_time)
                s0_perf.append(s0_time)
            bar_we.append(we_perf)
            bar_s0.append(s0_perf)

            mean_diff = fitness_mean_we - fitness_mean_s0

            try:
                cross_over_ind = np.where(mean_diff >= 0)[0][0]
                ax[ii].axvline(cross_over_ind, color=[0, 0, 0], linestyle=':')
            except:
                cross_over_ind = np.nan
            tteq_rob.append(cross_over_ind)
            ax[ii].xaxis.label.set_size(15)
            ax[ii].yaxis.label.set_size(15)
            ax[ii].set_title(skill_title[ii], fontsize=20)
            # ax[ii].legend()
            ax[ii].set_xlabel('time [min]')
            ax[ii].set_ylabel('cm/s' if skill == 'gait' else 'rad/s')
        tteq.append(tteq_rob)
        figure.tight_layout()
        figure.savefig(f"{results_dir}/{name}.pdf")

        plt.close(figure)
        fig = plt.figure(figsize=(4, 5))
        perf_perc = np.linspace(25, 25 * len(bar_we[0]), len(bar_we[0])).astype(int)
        width = 7.5
        bar_bot = np.cumsum(bar_we, axis=0)
        we_bar_list = [plt.bar(perf_perc, bar_we[0], align='edge', width=-width, label=f'WO: {skill_title[0]}'),
                       plt.bar(perf_perc, bar_we[1], align='edge', width=-width, bottom=bar_bot[0],
                               label=f'WO: {skill_title[1]}'),
                       plt.bar(perf_perc, bar_we[2], align='edge', width=-width, bottom=bar_bot[1],
                               label=f'WO: {skill_title[2]}')]

        cd_bar_list = [plt.bar(perf_perc, np.nan_to_num(np.nanmean(bar_s0, axis=0)), align='edge', width=width,
                               label=r'ISO$_{\Sigma}$')]

        plt.legend(fontsize=15)
        plt.grid(True)
        plt.xlabel('Normalized performance', fontsize=15)
        plt.xticks(perf_perc)
        # plt.yscale('log')
        plt.ylim([0, 900])
        plt.ylabel('Time in minutes', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{name}_perf.pdf")
        perc_lvl = 1
        print(f'{name} {np.sum(bar_we, axis=0)[perc_lvl]} vs {np.nanmean(bar_s0, axis=0)[perc_lvl]}:\n'
              f'\t {skill_title[0]} max: {np.round(max_s0[0], 3)}, {np.round(max_we[0], 3)} | tteq: {tteq_rob[0]}\n'
              f'\t {skill_title[1]} max: {np.round(max_s0[1], 3)}, {np.round(max_we[1], 3)} | tteq: {tteq_rob[1]}\n'
              f'\t {skill_title[2]} max: {np.round(max_s0[2], 3)}, {np.round(max_we[2], 3)} | tteq: {tteq_rob[2]}')

        print("FINISHED")