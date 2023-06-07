"""
Real world: Initial State Optimisation (ISO)
    Runs an ISO experiment (as presented in the paper) for given robot, and stores the data in ./results/REAL/ISO

How to use:
    !! Make sure mss is set up correctly !! - https://github.com/fudavd/multi-stitch-stream
    From the root directory you can run from terminal
    python ./exp_real/ISO.py --robot <robot name>
"""
import os
import sys
sys.path.extend([os.getcwd()])
import gc
import os.path
import time
import argparse
from typing import Dict

import numpy as np

from thirdparty.mss.src.Experiments import MotionCapture
from thirdparty.mss.src.Experiments.Robots import create_default_robot, show_grid_map
from thirdparty.mss.src.Experiments.Controllers import CPG
from thirdparty.mss.src.Experiments.Fitnesses import real_abs_dist, signed_rot
from thirdparty.mss.src.VideoStream import ExperimentStream
from thirdparty.mss.src.utils.Measures import find_closest
import logging
from revolve2.core.rpi_controller_remote import connect

argParser = argparse.ArgumentParser()
argParser.add_argument("-rob", "--robot", help="name of robot to be tested")

configuration_file: Dict = {
    'n_reps': 3,
    'n_trials': 5,
    'skills': ["gait", "rot_l", "rot_r"],
    'results_dir': './results/REAL/ISO',
    'robot': 'spider',
    'show_stream': False,
    'trial_time': 120,
    'window_time': 60,
    'hat_version': 'v1',
}


async def main(params, args) -> None:
    n_reps = params['n_reps']
    n_trials = params['n_trials']
    trial_time = params['trial_time']
    window_time = params['window_time']
    show_stream = params['show_stream']
    hat_version = params['hat_version']
    robot_name = args.robot
    error = False
    print(f'start experiment {robot_name}: {n_reps}x{n_trials} runs of {trial_time}s')
    robot_ip = np.loadtxt(f"./thirdparty/mss/secret/{robot_name}_ip.txt")
    for rep in range(n_reps):
        results_dir = f'{params["results_dir"]}/{robot_name}/{robot_name}_n{n_trials}_{trial_time}.{rep}'

        body, network_struct = create_default_robot(robot_name, random_seed=np.random.randint(1000))
        show_grid_map(body, robot_name, hat_version)

        matrix_weights = np.random.uniform(-1, 1, network_struct.num_params)
        weight_mat = network_struct.make_weight_matrix_from_params(matrix_weights)

        n_servos = int(network_struct.num_states / 2)

        if os.path.isfile(results_dir + '/weight_mat.npy'):
            weight_mat = np.load(results_dir + '/weight_mat.npy', allow_pickle=True)
        np.save(results_dir + '/weight_mat', weight_mat)
        initial_state = np.random.uniform(-1, 1, (n_servos * 2, n_trials))

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
        )
        c = []
        x = []
        f = []
        for pre_ind in range(run):
            c_trial = np.load(f'{results_dir}/{robot_name}_{pre_ind}/state.npy', allow_pickle=True)
            f_trial = np.load(f'{results_dir}/{robot_name}_{pre_ind}/fitnesses_trial.npy', allow_pickle=True)
            x_trial = np.load(f'{results_dir}/{robot_name}_{pre_ind}/x_trial.npy', allow_pickle=True)
            c.append(c_trial)
            f.append(np.array(f_trial).squeeze())
            x.append(x_trial)

        async with connect(robot_ip, "pi", "raspberry") as conn:
            print(f"Connection made with {robot_name}")
            with open("./thirdparty/mss/secret/cam_paths.txt", "r") as file:
                paths = file.read().splitlines()
            run = 0
            while run < n_trials and not error:
                experiment = ExperimentStream.ExperimentStream(paths, show_stream=show_stream,
                                                               output_dir=f'./{params["results_dir"]}/{robot_name}')
                print(f"Set new brain for run {run + 1}/{n_trials}")
                brain = CPG(n_servos, weight_mat, initial_state[:, run])
                config = brain.create_config(hat_version)

                capture = MotionCapture.MotionCaptureRobot(f'{robot_name}_{run}', ["red", "green"],
                                                           return_img=show_stream)
                experiment.start_experiment([capture.capture_aruco])
                robot_controller = asyncio.create_task(conn.run_controller(config, trial_time))
                experiment_run = asyncio.create_task(experiment.stream())
                tasks = [experiment_run, robot_controller]

                finished, unfinished = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                time.sleep(0.1)
                for task in finished:
                    print(task)
                    start_time, log = robot_controller.result()

                experiment.close_stream()
                await experiment_run

                # save results: capture_data, controller_data, start_data
                time.sleep(0.1)
                # capture.post_process_img_buffer(results_dir)
                capture.save_results(results_dir)
                np.save(f'{results_dir}/{robot_name}_{run}/state_con', log)
                np.save(f'{results_dir}/{robot_name}_{run}/start_con', start_time.timestamp())

                state_con = np.empty((0, len(log[0]['serialized_controller']['state'])))
                t_con = np.empty((0, 1))
                for sample in log:
                    t_con = np.vstack((t_con, sample['timestamp']))
                    state_con = np.vstack((state_con, sample['serialized_controller']['state']))

                capture_t = np.array(capture.t) - start_time.timestamp()
                capture_state = capture.robot_states
                capture_state = capture_state[(0 < capture_t) & (capture_t <= trial_time), :]
                capture_t = capture_t[(0 < capture_t) & (capture_t <= trial_time)]

                control_t = (t_con.flatten() - t_con[0, 0]) / 1000
                index = find_closest(control_t, capture_t)
                control_state = state_con[index]

                index = capture_t < (trial_time - window_time)

                x.append(control_state[index])

                n_samples = index.sum()
                f_trial = []
                for ind in range(n_samples):
                    t_rel = capture_t[ind]
                    window_idx = (t_rel <= capture_t) & (capture_t <= t_rel + window_time)
                    f_dist = real_abs_dist(capture_state[window_idx][:, :2]).squeeze()
                    f_angle = signed_rot(capture_state[window_idx][:, 2:])
                    fitnesses = np.array([f_dist, f_angle, -f_angle])
                    if np.isnan(f_angle):
                        fitnesses[1:] = -np.inf
                    f_trial.append(fitnesses)
                try:
                    print(f'Run {run}:', np.nanmax(f_trial, axis=0))
                except Exception as e:
                    print(e.with_traceback())
                f.append(np.array(f_trial).squeeze())
                c.append(capture_state)
                np.save(f'{results_dir}/{robot_name}_{run}/fitnesses_trial', f_trial)
                np.save(f'{results_dir}/{robot_name}_{run}/x_trial', control_state[index])
                run += 1

                capture.clear_buffer()
                del capture
                del experiment
                gc.collect()
                await asyncio.sleep(1)
        print(f"finished repetition {rep+1}, saving results")
        np.save(f'{results_dir}/capture_full', np.array(np.vstack(c)))
        np.save(f'{results_dir}/fitness_full', np.array(np.vstack(f)))
        np.save(f'{results_dir}/x_full', np.array(np.vstack(x)))
    print("FINISHED")


if __name__ == "__main__":
    import asyncio

    args = argParser.parse_args()
    parameters = configuration_file
    asyncio.run(main(parameters, args))
