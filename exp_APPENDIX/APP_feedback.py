"""
Isaac Gym: Initial State Optimisation (ISO)
    Runs a standard ISO experiment (as presented in paper) for a given robot, and stores the data in ./results/SIM/ISO

How to use:
    From the root directory you can run from terminal
    python ./exp_sim/SIM_ISO.py --robot <robot name>
"""
import copy

from matplotlib import pyplot as plt
from numpy.random import default_rng

from isaacgym import gymapi
from isaacgym import gymutil

from typing import Dict
import numpy as np

import os
import tempfile
from pyrr import Quaternion, Vector3

from thirdparty.mss.src.Experiments.Controllers import CPG_feedback
from exp_sim.sim_utils import set_controller, save_states, update_robots, fitness
from exp_APPENDIX.APP_ISO_multi_skill import generate_data
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
    'n_reps': 1,
    'skills': ["gait", "rot_l", "rot_r"],
    'results_dir': './results/APPENDIX/feedback',
    'robot': 'spider',
    'controller_update_time': 1 / 10,
    'trial_time': 600,
    'learner_params': None
}

def simulate_robot(params, args):
    results_dir = params['results_dir']
    skills = params['skills']
    robot_name = params['robot']
    controller_update_time = params['controller_update_time']
    trial_time = params['trial_time']
    params_learner = params['learner_params']
    window_time = params_learner['window_time']
    assert params['learner_params'] is not None, "Expected RevDE configuration"

    rep = 0
    initial_states = []
    for skill in skills:
        retest_dir = f'{results_dir}/{robot_name}/{skill}/{robot_name}{rep}/'
        weight_mat = np.load(f'{retest_dir}/weights.npy', allow_pickle=True)
        initial_states.append(np.load(f'{retest_dir}/x_best.npy', allow_pickle=True))

    n_servos = int(weight_mat.shape[0] / 2)
    CPG = CPG_feedback(n_servos, weight_mat, initial_states, window_time=window_time, cooldown_time=10, dt=controller_update_time)

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
    asset_root = "/home/fuda/PycharmProjects/multi-skill-learning/thirdparty/isaacgym/assets/"
    urdf_file = f"urdf/models/rg_robot/{robot_name}.urdf"

    print("Creating %d environments" % num_envs)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.04)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 0.707107)

    robot_handles = []
    envs = []

    target_ang = rng.random() * 2 * np.pi
    target_point = np.array([np.cos(target_ang), np.sin(target_ang), 0])*4
    origin = np.array([0, 0, 0])

    if not args.headless:
        num_per_row = int(np.floor(np.sqrt(num_envs)))
        # Create viewer
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        # Point camera at environments
        cam_pos = gymapi.Vec3(-target_point[0], -target_point[1], 4.0)
        cam_target = gymapi.Vec3(target_point[0], target_point[1], 0)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # create env
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        print("Loading asset '%s' from '%s', #%i" % (urdf_file, asset_root, i))
        robot_asset = gym.load_asset(
            sim, asset_root, urdf_file, asset_options)

        # add robot
        robot_handle = gym.create_actor(env, robot_asset, pose, f"robot #{i}", 1, 0)
        robot_handles.append(robot_handle)

    light_options = gymapi.AssetOptions()
    light_options.fix_base_link = True
    light_options.flip_visual_attachments = True
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*target_point)
    ball_asset = gym.create_sphere(sim, 0.25, light_options)
    ahandle = gym.create_actor(env, ball_asset, pose, None, 2, 0)
    color = gymapi.Vec3(0.1, 1.0, 0.1)
    gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL, color)
    gym.set_light_parameters(sim, 0, gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(0.0, 0.0, 1.0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(0.1, 0.1, 0.1),
                             gymapi.Vec3(0.0, 0.0, -1.0))

    pose.p = gymapi.Vec3(*origin)
    ball_asset = gym.create_sphere(sim, 0.05, light_options)
    ahandle = gym.create_actor(env, ball_asset, pose, None, 2, 0)
    color = gymapi.Vec3(0.0, 0.0, 0.8)
    gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL, color)
    gym.set_light_parameters(sim, 0, gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(0.0, 0.0, 1.0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(0.1, 0.1, 0.1),
                             gymapi.Vec3(0.0, 0.0, -1.0))

    # # get joint limits and ranges for robot
    props = gym.get_actor_dof_properties(envs[0], robot_handles[0])

    # Give a desired velocity to drive
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(1000.0)
    props["damping"].fill(600.0)
    for i in range(num_envs):
        gym.set_actor_dof_properties(envs[i], robot_handles[i], props)

    controllers = [CPG]
    state_buffer = np.zeros((num_envs, 6, round(trial_time / controller_update_time)))
    # state_buffer = np.zeros((num_envs, 6, 10000000))
    t_k = 0

    frame = 0
    proj_index = 0
    gait_ang = 0
    # %% Simulate
    cur_skill = 0
    cur_index = 0
    skills_list = ['→', '⟳', '⟲']
    cur_skill_buffer = ['→']
    skill_switch_time_buffer = [0]
    skill_flip_buffer = []
    skill_flip_time_buffer = []
    skill_buffer = [cur_skill]
    index_buffer = [cur_index]
    heading_buffer = []
    heading_error_buffer = []
    prev_state = origin.astype(np.float32)
    trace_buffer = np.zeros_like(state_buffer.T.squeeze())
    axes_geom = gymutil.AxesGeometry(0.2)
    # Get input actions from the viewer and handle them appropriately
    # # Step rendering
    t=0
    while t_k<round(trial_time/controller_update_time):
        t = gym.get_sim_time(sim)

        if round(t % controller_update_time, 2) == 0.0:
            # if True:
            state_buffer = save_states(gym, t_k, envs, robot_handles, state_buffer)
            cur_pos = copy.deepcopy(state_buffer[:, :3, t_k].squeeze())
            cur_line = np.array([prev_state[0], prev_state[1], 0.001, cur_pos[0], cur_pos[1], 0.001])
            trace_buffer[t_k, :] = cur_line
            prev_state = cur_pos
            for controller in controllers:
                heading_ang = state_buffer[:, -1, t_k] + gait_ang
                target_dir = target_point - state_buffer[:, :3, t_k]
                heading_error = -np.pi + (heading_ang - np.arctan2(target_dir[:, 1], target_dir[:, 0]) + np.pi)%(2*np.pi)

                if t_k < 30/controller_update_time:
                    gait_dir = np.mean(state_buffer[:, :3, :max(1, t_k-int(20/controller_update_time))], axis=2)
                    gait_ang = np.arctan2(gait_dir[:, 1], gait_dir[:, 0])%(2*np.pi)

                controller.heading_error = heading_error * (t_k >= 30/controller_update_time)
            heading_buffer.append(heading_ang)
            heading_error_buffer.append(heading_error)

            update_robots(gym, controllers, robot_handles, envs, controller_update_time)
            cur_index += 1

            if CPG.switch:
                print("SWITCH", (skills_list[CPG.prev_skill], CPG.prev_ind), (skills_list[CPG.cur_skill], CPG.cur_ind))
                skill_switch_time_buffer.append(t + 0.5 * sim_params.dt)
                cur_skill_buffer.append(skills_list[CPG.cur_skill])
                if CPG.cur_skill != CPG.prev_skill:
                    skill_flip_buffer.append(cur_skill_buffer.pop())
                    skill_flip_time_buffer.append(skill_switch_time_buffer.pop())

            if cur_skill != controller.cur_skill or cur_index != controller.cur_ind:
                skill_buffer.append(cur_skill)
                index_buffer.append(cur_index)
                skill_buffer.append(controller.cur_skill)
                index_buffer.append(controller.cur_ind)
                cur_skill = controller.cur_skill
                cur_index = controller.cur_ind
            t_k += 1


            if viewer:
                frame += 1
                gym.write_viewer_image_to_file(viewer, f'{results_dir}/viewer/{frame}.png')
            # if t_k % 60 == 0:
            #     print(skills[cur_skill])
            #     print(np.array((skill_buffer, index_buffer)))
            # print(index_buffer)

        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.clear_lines(viewer)

        core_state = gym.get_actor_rigid_body_states(envs[0], robot_handles[0], gymapi.STATE_NONE)
        robot_pose = gymapi.Transform(core_state["pose"]['p'][0], core_state["pose"]["r"][0])
        gymutil.draw_lines(axes_geom, gym, viewer, envs[0], robot_pose)

        colors = np.array([[1.0, 1.0, 1.0]]*t_k, dtype=np.float32)
        gym.add_lines(viewer, envs[0], t_k, trace_buffer[:t_k].astype(np.float32), colors)
        # Get input actions from the viewer and handle them appropriately
        # # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)

        if np.abs(target_dir).sum() < 0.50:
            break

    np.save(f'{results_dir}/data/target_point.npy', target_point)
    np.save(f'{results_dir}/data/robot_states.npy', state_buffer.squeeze()[:,:t_k])
    np.save(f'{results_dir}/data/CPG_states.npy', controller.y[:,:t_k])
    np.save(f'{results_dir}/data/heading.npy', heading_buffer)
    np.save(f'{results_dir}/data/heading_error.npy', heading_error_buffer)

    np.save(f'{results_dir}/data/transition/skill_switch_time_buffer.npy', skill_switch_time_buffer)
    np.save(f'{results_dir}/data/transition/cur_skill_buffer.npy', cur_skill_buffer)
    np.save(f'{results_dir}/data/transition/skill_flip_buffer.npy', skill_flip_buffer)
    np.save(f'{results_dir}/data/transition/skill_flip_time_buffer.npy', skill_flip_time_buffer)


def plot_perturbation_me(robot='spider'):
    import numpy as np
    from thirdparty.mss.src.Experiments.Controllers import CPG_feedback

    window_time = 60
    dt = 0.1
    robot_name = robot
    rep = 0
    initial_states = []
    results_dir = './results/APPENDIX/feedback'
    skills = ['gait', 'rot_l', 'rot_r']
    for skill in skills:
        retest_dir = f'{results_dir}/{robot_name}/{skill}/{robot_name}{rep}/'
        weight_mat = np.load(f'{retest_dir}/weights.npy', allow_pickle=True)
        initial_states.append(np.load(f'{retest_dir}/x_best.npy', allow_pickle=True))

    n_servos = int(weight_mat.shape[0] / 2)
    CPG = CPG_feedback(n_servos, weight_mat, initial_states, window_time, cooldown_time=15, dt=dt)

    import matplotlib.pyplot as plt
    skills = ['→', '⟲', '⟳']
    test_time = 60
    skill_switch_time = [0]
    cur_skill = ['→']
    skill_flip = []
    skill_flip_time = []
    for t in range(int(test_time/dt)):
        if t >= 35/dt - 2:
            CPG.heading_error = -1
        CPG.get_dof_targets()
        if CPG.switch:
            print("SWITCH", (skills[CPG.prev_skill], CPG.prev_ind), (skills[CPG.cur_skill], CPG.cur_ind))
            skill_switch_time.append((t+0.5)*dt)
            cur_skill.append(skills[CPG.cur_skill])
            if CPG.cur_skill != CPG.prev_skill:
                skill_flip.append(cur_skill.pop())
                skill_flip_time.append(skill_switch_time.pop())

    # fig, ax = plt.subplots(2)
    # x_neurons = CPG.skills[0, :n_servos, :]
    # y_neurons = CPG.skills[0, n_servos:, :]
    # ax[0].plot(x_neurons.T)
    # ax[1].plot(y_neurons.T)
    # fig.show()
    time = np.arange(int(test_time/dt)) * dt
    for i in range(n_servos):
        fig, ax = plt.subplots(2, figsize=(6, 4))
        x_neurons = CPG.skills[0, i, :]
        x_reset = CPG.y[i, :-1]
        y_neurons = CPG.skills[0, n_servos + i, :]
        y_reset = CPG.y[n_servos + i, :-1]
        ax[0].plot(time, x_neurons, linewidth=2)
        ax[0].plot(time, x_reset, linewidth=2)
        ax[1].plot(time, y_neurons, linewidth=2)
        ax[1].plot(time, y_reset, linewidth=2)

        for skill_time, skill in zip(skill_switch_time, cur_skill):
            ax[0].text(skill_time + dt, 1.1, skill, fontsize=12)
            ax[1].text(skill_time + dt, 1.1, skill, fontsize=12)

        ax[0].vlines(skill_switch_time, -2, 2, linestyles=':', color='k', linewidth=1)
        ax[1].vlines(skill_switch_time, -2, 2, linestyles=':', color='k', linewidth=1)

        for skill_time, skill in zip(skill_flip_time, skill_flip):
            ax[0].text(skill_time + dt, 1.1, skill, fontsize=12)
            ax[1].text(skill_time + dt, 1.1, skill, fontsize=12)
            ax[0].vlines(skill_time, -2, 2, linestyles='--', color='r', linewidth=1.5)
            ax[1].vlines(skill_time, -2, 2, linestyles='--', color='r', linewidth=1.5)
        ax[0].set_ylim([-1.5, 1.5])
        ax[1].set_ylim([-1.5, 1.5])

        ax[0].xaxis.label.set_size(15)
        ax[0].yaxis.label.set_size(15)
        ax[1].xaxis.label.set_size(15)
        ax[1].yaxis.label.set_size(15)
        ax[0].set_title("Skill switching behaviour", fontsize=16)
        ax[0].set_ylabel('x-neuron', fontsize=12)
        ax[1].set_ylabel('y-neuron', fontsize=12)
        ax[1].set_xlabel('Time (s)', fontsize=12)
        fig.tight_layout()
        fig.savefig(f'{results_dir}/{robot_name}{i}.pdf')
        # fig.show()

def plot_results():
    import numpy as np
    import matplotlib.pyplot as plt

    dt = 0.1
    window_time = 60
    window_size = int(window_time/dt)
    neuron = 0
    results_dir = './results/APPENDIX/feedback'

    target_point = np.load(f'{results_dir}/data/target_point.npy', allow_pickle=True)
    state_buffer = np.load(f'{results_dir}/data/robot_states.npy', allow_pickle=True)
    controller_buffer = np.load(f'{results_dir}/data/CPG_states.npy', allow_pickle=True)
    heading_buffer = np.load(f'{results_dir}/data/heading.npy', allow_pickle=True)
    heading_error_buffer = np.load(f'{results_dir}/data/heading_error.npy', allow_pickle=True)
    skill_switch_time_buffer = np.load(f'{results_dir}/data/transition/skill_switch_time_buffer.npy', allow_pickle=True)
    cur_skill_buffer = np.load(f'{results_dir}/data/transition/cur_skill_buffer.npy', allow_pickle=True)
    skill_flip_buffer = np.load(f'{results_dir}/data/transition/skill_flip_buffer.npy', allow_pickle=True)
    skill_flip_time_buffer = np.load(f'{results_dir}/data/transition/skill_flip_time_buffer.npy', allow_pickle=True)

    n_servos = int(controller_buffer.shape[0] / 2)
    x_neuron = controller_buffer[neuron]
    y_neuron = controller_buffer[int(n_servos/2) + neuron]
    time = np.arange(len(heading_buffer)) * dt
    for ind in range(0,600+1):
        fig, ax = plt.subplots(2, figsize=(6, 4))
        local_time = time[max(0, ind-window_size + 1):ind+1]

        ax[0].plot(local_time, x_neuron[max(0, ind-window_size + 1):ind+1], color='orange', linewidth=2)
        ax[1].plot(local_time, y_neuron[max(0, ind-window_size + 1):ind+1], color='orange', linewidth=2)
        ax[0].vlines(local_time[-1], -2, 2, linestyles='-', color='k', linewidth=1)
        ax[1].vlines(local_time[-1], -2, 2, linestyles='-', color='k', linewidth=1)
        ax[0].scatter(local_time[-1], x_neuron[ind], color='k', zorder=4, linewidth=2)
        ax[1].scatter(local_time[-1], y_neuron[ind], color='k', zorder=4, linewidth=2)

        skills_switched_ind = (local_time[:ind+1].min() <= skill_switch_time_buffer) * (skill_switch_time_buffer <= local_time[:ind+1].max())
        skill_switch_time = skill_switch_time_buffer[skills_switched_ind]
        cur_skill = cur_skill_buffer[skills_switched_ind]
        for skill_time, skill in zip(skill_switch_time, cur_skill):
            ax[0].text(skill_time + dt, 1.1, skill, fontsize=12)
            ax[1].text(skill_time + dt, 1.1, skill, fontsize=12)

        ax[0].vlines(skill_switch_time, -2, 2, linestyles=':', color='k', linewidth=1)
        ax[1].vlines(skill_switch_time, -2, 2, linestyles=':', color='k', linewidth=1)

        skills_flip_ind = (local_time[:ind+1].min() <= skill_flip_time_buffer) * (skill_flip_time_buffer <= local_time[:ind+1].max())
        skill_flip_time = skill_flip_time_buffer[skills_flip_ind]
        skill_flip = skill_flip_buffer[skills_flip_ind]
        for skill_time, skill in zip(skill_flip_time, skill_flip):
            ax[0].text(skill_time + dt, 1.1, skill, fontsize=12)
            ax[1].text(skill_time + dt, 1.1, skill, fontsize=12)
            ax[0].vlines(skill_time, -2, 2, linestyles='--', color='r', linewidth=1.5)
            ax[1].vlines(skill_time, -2, 2, linestyles='--', color='r', linewidth=1.5)
        ax[0].set_ylim([-1.5, 1.5])
        ax[1].set_ylim([-1.5, 1.5])
        ax[0].set_xlim([local_time.min(), max(60, local_time.max()) + 5])
        ax[1].set_xlim([local_time.min(), max(60, local_time.max()) + 5])

        ax[0].xaxis.label.set_size(15)
        ax[0].yaxis.label.set_size(15)
        ax[1].xaxis.label.set_size(15)
        ax[1].yaxis.label.set_size(15)
        ax[0].set_title("Skill switching behaviour", fontsize=16)
        ax[0].set_ylabel('x-neuron', fontsize=12)
        ax[1].set_ylabel('y-neuron', fontsize=12)
        ax[1].set_xlabel('Time (s)', fontsize=12)
        # fig.tight_layout()
        fig.savefig(f'{results_dir}/data/video/neurons/{ind+1}.png')
        plt.close()

    last_skill = None
    for ind in range(len(heading_buffer)):
        fig, ax = plt.subplots(1, figsize=(4, 4))
        ax.scatter(0, 0, color=(0.1, 0.1, 1.0), linewidth=2)
        ax.scatter(target_point[0], target_point[1], color=(0.1, 1.0, 0.1), linewidth=2, s=2**10)
        ax.text(0, 0, '→', fontsize=12, zorder=4)
        local_time = time[:ind+1]
        local_ind = np.arange(len(heading_buffer))[:ind + 1]
        x_pos = state_buffer[0, local_ind]
        y_pos = state_buffer[1, local_ind]

        ax.plot(x_pos, y_pos, color=(0.3,0.3,0.3), linewidth=2)
        ax.scatter(x_pos[-1], y_pos[-1], color='k', zorder=4, linewidth=2)

        skills_flip_ind = skill_flip_time_buffer <= local_time[:ind + 1].max()
        skill_flip_ind = skill_flip_time_buffer[skills_flip_ind]/dt
        skill_flip = skill_flip_buffer[skills_flip_ind]
        for skill_ind, skill in zip(skill_flip_ind, skill_flip):
            if last_skill == skill:
                pass
            else:
                curr_pos = [state_buffer[0, int(skill_ind)], state_buffer[1, int(skill_ind)]]
                ax.scatter(curr_pos[0], curr_pos[1], color=(1.0, 0.1, 0.1), s=1.5
                           , zorder=1)
                ax.text(curr_pos[0], curr_pos[1], skill, fontsize=12)
                heading_vec_x = [curr_pos[0], curr_pos[0] + 0.5*np.cos(heading_buffer[int(skill_ind)][0])]
                heading_vec_y = [curr_pos[1], curr_pos[1] + 0.5*np.sin(heading_buffer[int(skill_ind)][0])]
                ax.plot(heading_vec_x, heading_vec_y, color=(1.0, 0.1, 0.1), linewidth=1)
            last_skill = skill
        last_skill = None

        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])

        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.set_title("Robot behaviour", fontsize=16)
        ax.set_ylabel('y-position', fontsize=12)
        ax.set_xlabel('x-position', fontsize=12)
        fig.tight_layout()
        fig.savefig(f'{results_dir}/data/video/robot_state/{ind + 1}.png')
        plt.close()
        # fig.show()

if __name__ == "__main__":
    robot = 'spider'
    results_dir = './results/APPENDIX/feedback'
    # Parse arguments
    # if not os.path.exists(f'{results_dir}/spider0.pdf'):
    #     plot_perturbation_me(robot)
    # args = gymutil.parse_arguments(description="Loading and testing", headless=True,
    #                                custom_parameters=[{"name": "--robot",
    #                                                    "type": str}])
    #
    # params = configuration_file
    # params['learner_params'] = learner_params
    #
    # params['robot'] = robot
    # params['skills'] = ['gait', 'rot_l', 'rot_r']
    # generate_data(params, [params['robot']])
    # params['learner_params']['pop_size'] = 1
    # simulate_robot(params, args)
    plot_results()