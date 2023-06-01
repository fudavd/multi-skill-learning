"""
Loading and testing
"""
import copy
import os
import random

from numpy.random import default_rng

from isaacgym import gymapi
from isaacgym import gymutil

from typing import Dict
import numpy as np

import os
import tempfile
from pyrr import Quaternion, Vector3

from sim_utils import set_controller, save_states, update_robots, fitness
from utils.Learners import RandomSampling
from thirdparty.revolve2.standard_resources.revolve2.standard_resources import modular_robots
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics.running import PosedActor
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.modular_robot.brains import make_cpg_network_structure_neighbour as mkcpg, \
    BrainCpgNetworkNeighbourRandom
import time

rng = default_rng()

learner_params = {'evaluate_objective_type': 'full', 'max_gen': 1, 'pop_size': 150, 'window_time': 60,
                  'type': 'rs', 'sampling': np.random.uniform,
                  'genome_type': 'states'}

configuration_file: Dict = {
    'n_reps': 30,
    'skills': ["gait", "rot_l", "rot_r"],
    'results_dir': './results/SIM/WO',
    'robot': 'spider',
    'controller_update_time': 1 / 10,
    'trial_time': 120,
    'learner_params': None
}

def simulate_robot(params, args):
    n_reps = params['n_reps']
    results_dir = params['results_dir']
    skills = params['skills']
    robot_name = params['robot']
    controller_update_time = params['controller_update_time']
    trial_time = params['trial_time']
    params_learner = params['learner_params']
    window_time = params_learner['window_time']
    assert params['learner_params'] is not None, "Expected RevDE configuration"

    def parse_trial(state_buffer_eps, window, controller_states, num_skills):
        rewards = np.empty((0, num_skills))
        n_start = 0
        for n_start in range(1, state_buffer_eps.shape[1] - window):
            trajectory = state_buffer_eps[:, n_start:n_start + window]
            fitness_rotl = fitness(trajectory, "rot_l")
            fitness_rotr = fitness(trajectory, "rot_r")
            gait = fitness(trajectory, "gait")
            rewards = np.vstack((rewards, [gait, fitness_rotl, fitness_rotr]))
        return controller_states[1:n_start + window + 1, :], np.unwrap(rewards, axis=0)

    def update_learner(learner, state_buffer, controllers_buffer, genomes, fitnesses, full_rewards, full_states, window_time):
        window_size = int(window_time / controller_update_time)
        total_max = np.array([-np.inf] * 3)
        fitness_vec = []
        mean_f = 0
        for i in range(len(learner.x_new)):
            controller_buffer = np.array(controllers_buffer[i])
            states, rewards = parse_trial(state_buffer[i, :, :], window_size, controller_buffer, num_skills)
            full_states = np.vstack((full_states, states[:-window_size, :]))
            full_rewards = np.vstack((full_rewards, rewards[:, :]))

            trial_max = np.max(rewards, axis=0)
            total_max = np.max((total_max, trial_max), axis=0)

            fitness_vec.append(-trial_max)
            fitnesses.append(trial_max)
            mean_f += trial_max / len(learner.x_new)
        learner.f = np.array(fitness_vec)
        learner.x = learner.x_new
        learner.x_new = np.array([])
        new_genomes = learner.get_new_genomes()
        return np.vstack((genomes, new_genomes)), fitnesses, full_rewards, full_states

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    if args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 15
        sim_params.flex.relaxation = 0.75
        sim_params.flex.warm_start = 0.8
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = True

    if args.headless:
        args.graphics_device_id = -1
        print("*** Did not create viewer")

    # %% Initialize gym
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # %% Initialize environment
    print("Initialize environment")
    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0
    gym.add_ground(sim, plane_params)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01

    # Set up the env grid
    num_envs = params_learner['pop_size']

    spacing = 1
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    num_per_row = num_envs
    if not args.headless:
        num_per_row = int(np.floor(np.sqrt(num_envs)))
        # Create viewer
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        # Point camera at environments
        cam_pos = gymapi.Vec3((num_per_row - 1) * spacing, -4.0, 4.0)
        cam_target = gymapi.Vec3((num_per_row - 1) * spacing, 3.0, 0)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
        # # subscribe to spacebar event for reset
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

    # %% Initialize robots: Robot
    print("Initialize Robot")

    body = modular_robots.get(robot_name)
    brain = BrainCpgNetworkNeighbourRandom(rng)
    robot = ModularRobot(body, brain)
    actor, controller = robot.make_actor_and_controller()
    posed_actor = PosedActor(
        actor,
        Vector3([0.0, 0.0, 0.1]),
        Quaternion(),
        [0.0 for _ in controller.get_dof_targets()],
    )
    botfile = tempfile.NamedTemporaryFile(
        mode="r+", delete=False, suffix=".urdf"
    )
    botfile.writelines(
        physbot_to_urdf(
            posed_actor.actor,
            robot_name,
            Vector3(),
            Quaternion(),
        )
    )
    botfile.close()
    asset_root = os.path.dirname(botfile.name)
    urdf_file = os.path.basename(botfile.name)

    _, dof_ids = body.to_actor()
    active_hinges_clean = body.find_active_hinges()
    active_hinge_map = {active_hinge.id: active_hinge for active_hinge in active_hinges_clean}
    active_hinges_sim = [active_hinge_map[id] for id in dof_ids]
    network_struct = mkcpg(active_hinges_sim)

    print("Creating %d environments" % num_envs)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.04)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 0.707107)

    robot_handles = []
    envs = []
    # create env
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        print("Loading asset '%s' from '%s', #%i" % (urdf_file, asset_root, i))
        robot_asset = gym.load_asset(
            sim, asset_root, urdf_file, asset_options)

        # add robot
        robot_handle = gym.create_actor(env, robot_asset, pose, f"robot #{i}")
        robot_handles.append(robot_handle)

    # # get joint limits and ranges for robot
    props = gym.get_actor_dof_properties(envs[0], robot_handles[0])

    # Give a desired velocity to drive
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(1000.0)
    props["damping"].fill(600.0)

    start_rep = time.time()
    rep_count = 0
    num_skills = len(skills)
    for rep in range(n_reps):
        finished = num_skills
        for ind, skill in enumerate(skills):
            learner_dir = f"./{results_dir}/{robot_name}/{skill}/{robot_name}{rep}"
            if os.path.isfile(f'{learner_dir}/fitnesses.npy'):
                print(f'Already learned {robot_name}{rep}: exp = {skill}')
                finished -= 1
                continue
            if not os.path.exists(learner_dir):
                os.makedirs(learner_dir)
        if finished == 0:
            print(f'Already learned {robot_name}{rep}: exp = {skills}')
            break

        _, controller = robot.make_actor_and_controller()
        rep_count += 1
        genomes = np.random.uniform(-1, 1, (num_envs, network_struct.num_states))
        controllers = []
        controllers_buffer = [None] * num_envs
        for i in range(num_envs):
            gym.set_actor_dof_properties(envs[i], robot_handles[i], props)
            state0 = genomes[i, :]
            controller_t = set_controller(robot, state0, network_struct, params_learner['genome_type'])
            controllers.append(controller_t)
            controllers_buffer[i] = [state0]

        # %% Initialize learner
        learner = RandomSampling(genomes, num_envs, (-1, 1), params_learner, output_dir=results_dir)

        state_buffer = np.zeros((num_envs, 6, round(trial_time / controller_update_time)))
        t_k = 0
        gen = 0
        start = True
        fitnesses = []
        full_states = np.empty((0, network_struct.num_states))
        full_rewards = np.empty((0, num_skills))
        # %% Simulate
        while gen < params_learner['max_gen']:
            t = gym.get_sim_time(sim)
            if round(t % controller_update_time, 2) == 0.0:
                if round(t, 2) % trial_time == 0.0 and not start:
                    genomes, fitnesses, full_rewards, full_states = update_learner(learner, state_buffer, controllers_buffer,
                                                                                   genomes, fitnesses,
                                                                                   full_rewards, full_states, window_time)
                    for ind in range(len(learner.f)):
                        genome = learner.x_new[ind, :]
                        new_controller = set_controller(robot, genome, network_struct, learner_params['genome_type'])
                        controllers[ind] = new_controller
                    t_k = 0
                    gen += 1
                    state_buffer = np.zeros((num_envs, 6, round(trial_time / controller_update_time)))
                start = False
                state_buffer = save_states(gym, t_k, envs, robot_handles, state_buffer)
                t_k += 1
                update_robots(gym, controllers, robot_handles, envs, controller_update_time)
                for i in range(num_envs):
                    controllers_buffer[i].append(controllers[i]._state)

            # Step the physics
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Get input actions from the viewer and handle them appropriately
            if not args.headless:
                # # Step rendering
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, False)

        print(f"Finished learning {skills} on {robot_name}: {rep}")
        best_ind = np.argmax(full_rewards, axis=0)
        best_fitness = np.diagonal(full_rewards[best_ind])
        best_genomes = full_states[best_ind]
        print(best_fitness)
        for ind, skill in enumerate(skills):
            learner_dir = f"./{results_dir}/{robot_name}/{skill}/{robot_name}{rep}"
            if not os.path.exists(learner_dir):
                os.makedirs(learner_dir)
            if os.path.isfile(f'{learner_dir}/fitnesses.npy'):
                print(f'Already learned {robot_name}{rep}: exp = {skill}')
                continue

            np.save(f'{learner_dir}/genomes.npy', full_states[:, ind])
            np.save(f'{learner_dir}/fitnesses.npy', full_rewards[:, ind])
            np.save(f'{learner_dir}/f_best.npy', best_fitness[ind])
            np.save(f'{learner_dir}/x_best.npy', best_genomes[ind, :])
            np.save(f'{learner_dir}/weights.npy', controller._weight_matrix)

    print(
        f'Learned {robot_name}, {skills.__iter__()} for {rep_count} repetitions: avg {(time.time() - start_rep) / 60 / max(1, rep_count)}min')

    print(f"destroying simulation: {(time.time() - start) / 3600}")
    gym.destroy_sim(sim)


if __name__ == "__main__":
    # Parse arguments
    args = gymutil.parse_arguments(description="Loading and testing", headless=True,
                                   custom_parameters=[{"name": "--robot",
                                                       "type": str}])

    params = configuration_file
    params['learner_params'] = learner_params
    if args.robot:
        params['robot'] = args.robot
    simulate_robot(params, args)


