"""
Retest learned skill for 60 seconds in the real world, and stores the data in ./results/REAL/...


How to use:
    !! Make sure mss is set up correctly !! - https://github.com/fudavd/multi-stitch-stream
    From the root directory you can run from terminal
    python ./exp_real/retest_CPG.py -dir <path/to/retest_dir>
"""
import os
import sys
sys.path.extend([os.getcwd()])
import gc
from typing import Dict

import numpy as np
import argparse
from revolve2.core.modular_robot import ModularRobot

from revolve2.core.modular_robot.brains import make_cpg_network_structure_neighbour as mkcpg, \
    BrainCpgNetworkNeighbourRandom

from ISO import configuration_file
from thirdparty.mss.src.Experiments import MotionCapture
from thirdparty.mss.src.Experiments.Controllers import CPG
from thirdparty.mss.src.Experiments.Fitnesses import real_abs_dist, signed_rot
from thirdparty.mss.src.Experiments.Robots import show_grid_map
from thirdparty.mss.src.VideoStream import ExperimentStream
from thirdparty.mss.src.utils.Measures import find_closest
from thirdparty.revolve2.standard_resources.revolve2.standard_resources import modular_robots
import logging
from revolve2.core.rpi_controller_remote import connect

from utils.utils import robot_names, optim_types, skill_set

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dir", help="path to robot folder to be retested")


async def main(params, args) -> None:
    results_dir = args.dir
    robot_name = [robot for robot in robot_names if robot in results_dir][0]
    opt_type = [optim_type for optim_type in optim_types if optim_type in results_dir][0]
    skills = params['skills']
    trial_time = params['window_time']
    show_stream = params['show_stream']
    hat_version = params['hat_version']
    body = modular_robots.get(robot_name)
    show_grid_map(body, robot_name, hat_version)
    robot_ip = np.loadtxt(f"./thirdparty/mss/secret/{robot_name}_ip.txt")

    if opt_type == 'ISO':
        weight_mat = np.load(f'{results_dir}/weight_mat.npy', allow_pickle=True)
        weight_mats = np.array([weight_mat, weight_mat, weight_mat, ])
        f = np.load(f'{results_dir}/fitness_full.npy', allow_pickle=True)
        x = np.load(f'{results_dir}/x_full.npy', allow_pickle=True)

        init_idx = np.nanargmax(f, axis=0)
        print("expected fitnesses: \n",
              f[init_idx])
        initial_state = x[init_idx, :]
    elif opt_type == 'WO':
        _, dof_ids = body.to_actor()
        from random import Random
        rng = Random()
        rng.seed(420)
        brain = BrainCpgNetworkNeighbourRandom(rng)
        robot = ModularRobot(body, brain)
        _, controller = robot.make_actor_and_controller()
        active_hinges_clean = body.find_active_hinges()
        active_hinge_map = {active_hinge.id: active_hinge for active_hinge in active_hinges_clean}
        active_hinges_sim = [active_hinge_map[id] for id in dof_ids]
        network_struct = mkcpg(active_hinges_sim)
        genomes = []
        weight_mats = []
        for skill in skills:
            genome = np.loadtxt(f'{results_dir}/weights_{skill}.txt', delimiter=', ')
            weight_mats.append(network_struct.make_connection_weights_matrix_from_params(genome))
            genomes.append(controller._state)
        initial_state = np.array(genomes)
        weight_mats = np.array(weight_mats)
    else:
        print("NON VALID")
        exit()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    async with connect(robot_ip, "pi", "raspberry") as conn:
        print(f"Connection made with {robot_name}")
        with open("./thirdparty/mss/secret/cam_paths.txt", "r") as file:
            paths = file.read().splitlines()

        skill_ind = [ind for ind, skill in enumerate(skill_set) if skill in skills]
        for ind in skill_ind:
            experiment = ExperimentStream.ExperimentStream(paths, show_stream=show_stream,
                                                           output_dir=results_dir)
            print(f"Test {results_dir} brain: skill {skills[ind]}")
            weight_mat = weight_mats[ind, :, :]
            n_servos = int(weight_mat.shape[0] / 2)
            brain = CPG(n_servos, weight_mat, initial_state[ind, :])
            config = brain.create_config(hat_version=hat_version)

            capture = MotionCapture.MotionCaptureRobot(f'{robot_name}_{skills[ind]}', ["red", "green"],
                                                       return_img=show_stream)
            experiment.start_experiment([capture.store_img])
            robot_controller = asyncio.create_task(conn.run_controller(config, trial_time))
            experiment_run = asyncio.create_task(experiment.stream())
            tasks = [experiment_run, robot_controller]
            # time.sleep(0.1)
            capture.clear_buffer()
            finished, unfinished = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in finished:
                print(task)
                start_time, log = robot_controller.result()

            experiment.close_stream()
            await experiment_run

            capture.post_process_img_buffer(results_dir)
            np.save(f'{results_dir}/{robot_name}_{skills[ind]}/state_con', log)
            np.save(f'{results_dir}/{robot_name}_{skills[ind]}/start_con', start_time.timestamp())

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

            index = capture_t < trial_time

            f_dist = real_abs_dist(capture_state[:, :2]).squeeze()
            f_angle = signed_rot(capture_state[:, 2:])
            fitnesses = np.array([f_dist, f_angle, -f_angle])
            print(f'Retest for {skills[ind]}:\n'
                  f'Fitnesses: {fitnesses}\n')

            np.save(f'{results_dir}/{robot_name}_{skills[ind]}/fitnesses_trial', fitnesses)
            np.save(f'{results_dir}/{robot_name}_{skills[ind]}/x_trial', control_state[index])

            capture.clear_buffer()
            del capture
            del experiment
            gc.collect()
            await asyncio.sleep(1)

    print("Finished")


if __name__ == "__main__":
    import asyncio
    args = argParser.parse_args()
    parameters = configuration_file
    if args.__contains__('dir'):
        asyncio.run(main(parameters, args))
    else:
        print("ERROR must provide directory argument: --dir")
