from numpy.random import default_rng
from isaacgym import gymapi
from typing import AnyStr
import numpy as np
from revolve2.core.modular_robot import ModularRobot


def get_robot_state(gym, env, robot):
    body_pos = gym.get_actor_rigid_body_states(env, robot, gymapi.STATE_ALL)["pose"]["p"][0]
    body_rot = gymapi.Quat.to_euler_zyx(
        gym.get_actor_rigid_body_states(env, robot, gymapi.STATE_ALL)["pose"]["r"][0])
    return np.append(
        [body_pos[0], body_pos[1], body_pos[2]],
        [body_rot[0], body_rot[1], body_rot[2]],
    )


def set_controller(robot: ModularRobot, genome, network_struct, genome_type):
    _, controller = robot.make_actor_and_controller()
    n_weights = network_struct.num_connections
    n_states = network_struct.num_states
    if genome_type.__contains__('weights'):
        controller._weight_matrix = network_struct.make_connection_weights_matrix_from_params(genome[:n_weights])
    if genome_type.__contains__('states'):
        controller._state = genome[-n_states:]
    return controller


def update_robots(gym, controllers, robot_handles, envs, controller_update_time):
    for i in range(len(robot_handles)):
        controller = controllers[i]
        robot_handle = robot_handles[i]
        env = envs[i]
        gym.set_actor_dof_position_targets(env, robot_handle, controller.get_dof_targets())
        controller.step(controller_update_time)


def fitness(trajectory: np.array, type: AnyStr = "gait"):
    original_pos = trajectory[:3, 0]
    current_pos = trajectory[:3, -1]
    travelled_distance = current_pos - original_pos
    travelled_distance[-1] = original_pos[-1]
    fitness_val = 0
    if type == "gait":
        fitness_val = np.linalg.norm(travelled_distance) * 100
    elif type == "rot_l":
        z_rot = np.unwrap(trajectory[-1, :], period=2.0 * np.pi)
        fitness_val = z_rot[-1] - z_rot[0]
    elif type == "rot_r":
        z_rot = np.unwrap(trajectory[-1, :], period=2.0 * np.pi)
        fitness_val = z_rot[0] - z_rot[-1]
    elif type == "side_l":
        z_rot = np.unwrap(trajectory[-1, :], period=2.0 * np.pi)
        deviation = np.matmul(z_rot.T,z_rot)/len(z_rot)
        side_dist = trajectory[0,-1] - original_pos[0]
        forward_dist = trajectory[1,-1] - original_pos[1]
        fitness_val = (side_dist-deviation - np.abs(forward_dist))*100
    elif type == "side_r":
        z_rot = np.unwrap(trajectory[-1, :], period=2.0 * np.pi)
        deviation = np.matmul(z_rot.T,z_rot)/len(z_rot)
        side_dist = trajectory[0,-1] - original_pos[0]
        forward_dist = trajectory[1,-1] - original_pos[1]
        fitness_val = (-side_dist - deviation - np.abs(forward_dist))*100
    elif type == "jump":
        fitness_val = np.max(trajectory[2, :])*60*100
    return fitness_val / 60


def save_states(gym, t_k, envs, robot_handles, state_buffer):
    for ind in range(len(robot_handles)):
        current_robot_handle = robot_handles[ind]
        state_t = get_robot_state(gym, envs[ind], current_robot_handle,)
        state_buffer[ind, :, t_k] = state_t
    return state_buffer
