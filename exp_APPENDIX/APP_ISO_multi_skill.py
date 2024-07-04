import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from utils.utils import search_file_list
from exp_sim.ISO import simulate_robot, learner_params, configuration_file
from isaacgym import gymutil
from cycler import cycler

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.prop_cycle'] = cycler(
    color=['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
           '#CFCFCF'])

robot_names = ['ant', 'gecko', 'babya', 'spider', 'blokky', 'salamander', 'stingray', 'garrix', 'insect', 'linkin',
               'longleg', 'penguin', 'pentapod', 'queen', 'squarish', 'babyb', 'tinlicker', 'turtle', 'ww', 'zappa']


def generate_data(configuration_file, robot_names=robot_names):
    args = gymutil.parse_arguments(description="Loading and testing", headless=True)
    args.headless = True
    for robot in robot_names:
        configuration_file['robot'] = robot
        simulate_robot(configuration_file, args)


if __name__ == "__main__":
    results_dir = os.path.join('results', 'APPENDIX/', 'multi_skill')
    skills = ['gait', 'rot_l', 'rot_r', 'jump', 'side_l', 'side_r']
    configuration_file['skills'] = skills
    configuration_file['learner_params'] = learner_params
    configuration_file['results_dir'] = results_dir

    # %% Generate data
    if not os.path.exists(f'{results_dir}/{robot_names[-1]}'):
        generate_data(configuration_file)
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
    rescale = [1, 1, 1, 1, 1, 1]
    skill_title = ['→', '⟲', '⟳', r'$^{^{^{ }}↑}$', r'$^{^{^{ }}\bigtriangleup}$', r'$^{^{^{ }}\bigtriangleup}$']
    bar_norm = [-np.inf]*len(skill_title)
    for name in robot_names:
        for ii, skill in enumerate(skills):
            folder_state0 = os.path.join(results_dir, name, skill)
            state0_list = search_file_list(folder_state0, 'f_best.npy')
            fitness_max_s0 = []
            for fitness_ref in state0_list:
                fitness_end = np.load(fitness_ref, allow_pickle=True)
                fitness_max_s0.append(fitness_end)
            if bar_norm[ii] < np.mean(fitness_max_s0):
                bar_norm[ii] = np.mean(fitness_max_s0)

    for name in robot_names:
        figure, ax = plt.subplots(1, figsize=(3.25, 3))
        bar_s0 = []
        se_s0 = []
        for ii, skill in enumerate(skills):
            folder_state0 = os.path.join(results_dir, name, skill)
            state0_list = search_file_list(folder_state0, 'f_best.npy')

            fitness_max_s0 = []
            for fitness_ref in state0_list:
                fitness_end = np.load(fitness_ref, allow_pickle=True)/bar_norm[ii]
                fitness_max_s0.append(fitness_end)

            bar_s0.append(np.mean(fitness_max_s0))
            error_norm = np.sqrt(len(state0_list))
            se_s0.append(np.std(fitness_max_s0)/error_norm*1.96)
            _state0 = np.load(fitness_ref.replace('f_best.npy', 'x_best.npy'), allow_pickle=True)

        ax.bar(np.arange(len(bar_s0)), bar_s0)
        ax.errorbar(np.arange(len(bar_s0)), bar_s0, se_s0, fmt='o', linewidth=2, capsize=6, color='k')

        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.grid(axis='y')
        ax.set_ylim([0.0, 1.175])
        ax.set_xticks(np.arange(len(bar_s0)), skill_title, fontsize=20)
        ax.set_ylabel('Performance', fontsize=20)

        figure.tight_layout()
        figure.canvas.draw()
        label = ax.get_xticklabels()[3]
        bbox = label.get_window_extent()  # Get the bounding box of the label in display coords
        bbox_transformed = bbox.transformed(figure.transFigure.inverted())  # Transform to figure fraction
        # Draw an underline using annotate
        ax.annotate('', xy=(bbox_transformed.x0-0.001, bbox_transformed.y0 + 0.03), xycoords='figure fraction',
                    xytext=(bbox_transformed.x1, bbox_transformed.y0 + 0.03), textcoords='figure fraction',
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", color='black', lw=1))

        label = ax.get_xticklabels()[4]
        bbox = label.get_window_extent()  # Get the bounding box of the label in display coords
        bbox_transformed = bbox.transformed(figure.transFigure.inverted())  # Transform to figure fraction
        # Draw an underline using annotate
        ax.annotate('', xy=(bbox_transformed.x0-0.001, bbox_transformed.y0 + 0.03), xycoords='figure fraction',
                    xytext=(bbox_transformed.x1, bbox_transformed.y0 + 0.03), textcoords='figure fraction',
                    annotation_clip=False, arrowprops=dict(arrowstyle="->", color='black', lw=1))

        label = ax.get_xticklabels()[5]
        bbox = label.get_window_extent()  # Get the bounding box of the label in display coords
        bbox_transformed = bbox.transformed(figure.transFigure.inverted())  # Transform to figure fraction
        # Draw an underline using annotate
        ax.annotate('', xy=(bbox_transformed.x0-0.001, bbox_transformed.y0 + 0.03), xycoords='figure fraction',
                    xytext=(bbox_transformed.x1, bbox_transformed.y0 + 0.03), textcoords='figure fraction',
                    annotation_clip=False, arrowprops=dict(arrowstyle="<-", color='black', lw=1))



        figure.savefig(f"{results_dir}/{name}.pdf")
        plt.close(figure)

        print(f"FINISHED {name} k={int(len(_state0)/2)}: {bar_s0}")
    print(bar_norm)