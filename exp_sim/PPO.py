"""
Isaac Gym: Initial State Optimisation (PPO)
    Runs SOTA PPO experiment (as presented in paper) for a given robot, and stores the data in ./results/SIM/PPO

How to use:
    From the root directory you can run from terminal
    python ./exp_sim/PPO.py --robot <robot name>
"""
from numpy.random import default_rng
import matplotlib.pyplot as plt

from isaacgym import gymapi
from isaacgym import gymutil

from typing import Dict, AnyStr
import numpy as np

import os
import tempfile
from pyrr import Quaternion, Vector3

from sim_utils import fitness, get_robot_state
from thirdparty.revolve2.standard_resources.revolve2.standard_resources import modular_robots
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics.running import PosedActor
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.modular_robot.brains import BrainCpgNetworkNeighbourRandom
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

rng = default_rng()
learner_params = {'window_time': 0.1,
                  'gamma': 0.99,
                  'gae-lambda': 0.95,
                  'num-minibatches': 100,
                  'update-epochs': 10,
                  'clip-coef': 0.2,
                  'ent-coef': 0.0,
                  'vf-coef': 2,
                  'max-grad-norm': True,
                  'lr': 3e-3,
                  'n_envs': 10,
                  }
# learner_params = {'window_time': 0.1,
#                   'gamma': 0.97,
#                   'gae-lambda': 0.95,
#                   'num-minibatches': 10,
#                   'update-epochs': 10,
#                   'clip-coef': 0.2,
#                   'ent coef': 0.0,
#                   'vf-coef': 2,
#                   'max-grad-norm': True,
#                   'lr': 3e-4,
#                   }


configuration_file: Dict = {
    'n_reps': 30,
    'skills': ["gait", "rot_l", "rot_r"],
    'results_dir': './results/SIM/PPO',
    'robot': 'spider',
    'controller_update_time': 1 / 10,
    'trial_time': 60,
    'learner_params': None
}


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, single_state_space, single_action_space):
        super().__init__()
        n_hid = 256
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(single_state_space).prod(), n_hid)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hid, n_hid)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hid, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(single_state_space).prod(), n_hid)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hid, n_hid)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hid, np.prod(single_action_space)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(single_action_space)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def simulate_robot(params, args):
    n_reps = params['n_reps']
    results_dir = params['results_dir']
    skills = params['skills']
    robot_name = params['robot']
    controller_update_time = params['controller_update_time']
    trial_time = params['trial_time']

    params_learner = params['learner_params']
    window_time = params_learner['window_time']
    gamma = params_learner['gamma']
    gae_lambda = params_learner['gae-lambda']
    num_minibatches = params_learner['num-minibatches']
    num_epochs = params_learner['update-epochs']
    ent_coef = params_learner['ent-coef']
    vf_coef = params_learner['vf-coef']
    clip_coef = params_learner['clip-coef']
    max_grad_norm = params_learner['max-grad-norm']
    learning_rate = params_learner['lr']
    num_envs = params_learner['n_envs']

    total_time = 300 * trial_time / num_envs
    num_steps = int(trial_time / controller_update_time)
    single_env_size = int(num_steps - (window_time / controller_update_time))
    batch_size = single_env_size * num_envs
    minibatch_size = int(batch_size // num_minibatches)
    [print(f"{key}: {value}") for key, value in params_learner.items()]
    print("(mini) batch size:", minibatch_size, batch_size)

    def parse_trial(obs_eps, window, skill: AnyStr = "gait"):
        trial_rewards = []
        for n_start in range(0, obs_eps.shape[0] - window):
            trajectory = obs_eps[n_start:n_start + window + 1, :]
            reward_skill = fitness(trajectory.T, skill) / window
            trial_rewards.append(reward_skill)
        trial_fitness = fitness(obs_eps.T, skill)
        # print(f"{skill} ({round(np.max(trial_rewards), 2)}): {trial_fitness.round(2)}, pos: {obs_eps[-1, :3].T - obs_eps[0, :3].T}, {np.linalg.norm(obs_eps[-1, :3].T - obs_eps[0, :3].T)*1.66666}")
        trial_rewards = np.array(trial_rewards)[:, np.newaxis] * abs(trial_fitness)
        return obs_eps[:len(trial_rewards), :], trial_rewards, trial_fitness

    def update_learner(obs_ep, actions_buffer, state_buffer, window_time, skill, eps, rg_thr):
        window_size = int(window_time / controller_update_time)
        trial_fitnesses = []
        for env_i in range(num_envs):
            obs_env = obs_ep[:, env_i]
            _obs_env, _rewards_env, trial_fitness = parse_trial(obs_env.squeeze().cpu().numpy(), window_size, skill)
            trial_fitnesses.append(trial_fitness)
            rewards[:single_env_size, env_i] = torch.tensor(_rewards_env.squeeze()).clone().detach()
        b_num_steps = len(_obs_env)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(state_buffer[b_num_steps, :, :]).reshape(1, -1)
            advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)
            lastgaelam = 0
            for time_eps in reversed(range(b_num_steps)):
                if time_eps == b_num_steps - 1:
                    nextnonterminal = 1.0
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[time_eps + 1]
                    nextvalues = values[time_eps + 1]
                delta = rewards[time_eps] + gamma * nextvalues * nextnonterminal - values[time_eps]
                advantages[time_eps] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_state = state_buffer[:b_num_steps].reshape((-1,) + single_state_space)
        b_logprobs = logprobs[:b_num_steps].reshape(-1)
        b_actions = actions_buffer[:b_num_steps].reshape((-1,) + single_action_space)
        b_advantages = advantages[:b_num_steps].reshape(-1)
        b_returns = returns[:b_num_steps].reshape(-1)
        b_values = values[:b_num_steps].reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_state[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)

                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                # v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        experiment_time = (eps * trial_time).__round__(2)
        global_step = int((experiment_time + trial_time) * num_envs / controller_update_time)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        for ind in range(num_envs):
            trial_n = eps * num_envs + ind + 1
            print(
                f"{f'{rep}.{trial_n}':<6} SPS: {int(global_step * num_envs / (time.time() - start_time))} | performance: {trial_fitnesses[ind].round(3):<5} {rewards[:single_env_size, ind].max():<5} t = {trial_n * trial_time}/{total_time * num_envs}s")
        writer.add_scalar("charts/SPS", int(global_step * num_envs / (time.time() - start_time)), global_step)
        return trial_fitnesses, eps, rg_thr

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
        # sim_params.physx.num_threads = 1
        sim_params.physx.use_gpu = True
        # sim_params.use_gpu_pipeline = True

    if args.headless:
        args.graphics_device_id = -1
        print("*** Did not create viewer")
    device = torch.device("cpu")
    device = torch.device("cuda" if (torch.cuda.is_available() and sim_params.physx.use_gpu) else "cpu")

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

    print("Creating %d environment" % num_envs)
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
    props = gym.get_actor_dof_properties(env, robot_handle)

    # Give a desired velocity to drive
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(1000.0)
    props["damping"].fill(600.0)

    for i in range(num_envs):
        gym.set_actor_dof_properties(envs[i], robot_handles[i], props)

    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    for skill in skills:
        rep = 0
        while rep < n_reps:
            start_time = time.time()
            learner_dir = f"./{results_dir}/{robot_name}/{skill}/{robot_name}{rep}"
            if not os.path.exists(learner_dir):
                os.makedirs(learner_dir)
            if os.path.isfile(f'{learner_dir}/rewards.pdf'):
                print(f'Already learned {robot_name}{rep}: exp = {skill}')
                rep += 1
                continue
            print(f'Learning gait for {robot_name}{rep}: exp = {skill}')
            writer = SummaryWriter(f"{learner_dir}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in params_learner.items()])),
            )

            # %% Initialize learner
            single_observation_space = (6,)
            single_state_space = (gym.get_actor_joint_count(env, robot_handle) * 2,)
            single_action_space = (gym.get_actor_joint_count(env, robot_handle),)
            agent = Agent(single_state_space, single_action_space).to(device)
            optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

            # ALGO Logic: Storage setup
            obs = torch.zeros((num_steps, num_envs) + single_observation_space).to(device)
            actions = torch.zeros((num_steps, num_envs) + single_action_space, dtype=torch.float).to(device)
            logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float).to(device)
            rewards = torch.zeros((num_steps, num_envs), dtype=torch.float).to(device)
            dones = torch.zeros((num_steps, num_envs), dtype=torch.float).to(device)
            values = torch.zeros((num_steps, num_envs), dtype=torch.float).to(device)
            advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

            state_buffer = torch.zeros((num_steps, num_envs) + single_state_space, dtype=torch.float).to(device)

            fitnesses = []
            t_k = 0
            t = -1
            eps = 0
            rg_thr = 1
            # %% Simulate
            num_updates = int(total_time * 60)
            experiment_time = 300 * 60
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
            while eps * num_envs * trial_time < experiment_time:
                t = gym.get_sim_time(sim)
                if round(t % controller_update_time, 2) == 0.0:
                    if t_k == len(obs):
                        fitness_eps, eps, rg_thr = update_learner(obs,
                                                                  actions,
                                                                  state_buffer,
                                                                  window_time,
                                                                  skill,
                                                                  eps, rg_thr)
                        fitnesses += fitness_eps
                        if not fitness_eps:
                            rep -= 1
                            break

                        gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
                        t_k = 0
                        state_buffer = torch.zeros((num_steps, num_envs) + single_state_space, dtype=torch.float).to(
                            device)
                        eps += 1

                    for ind in range(len(robot_handles)):
                        current_robot_handle = robot_handles[ind]

                        obs_t = torch.tensor(get_robot_state(gym, envs[ind], current_robot_handle, )).to(device)
                        obs[t_k, ind, :] = obs_t
                        state_pos = gym.get_actor_dof_states(envs[ind], current_robot_handle, gymapi.STATE_POS)['pos']
                        state_vel = gym.get_actor_dof_states(envs[ind], current_robot_handle, gymapi.STATE_VEL)['vel']
                        state_t = torch.tensor(np.hstack((state_pos, state_vel))[np.newaxis, :]).to(device)
                        state_buffer[t_k, ind, :] = state_t

                        with torch.no_grad():
                            action_t, logprob, _, value = agent.get_action_and_value(state_t)
                            values[t_k, ind] = value.flatten()
                        actions[t_k, ind] = action_t
                        logprobs[t_k, ind] = logprob
                        dones[t_k, ind] = False

                        gym.set_actor_dof_position_targets(envs[ind], current_robot_handle,
                                                           action_t.flatten().cpu().numpy())
                    t_k += 1
                # Step the physics
                gym.simulate(sim)
                gym.fetch_results(sim, True)

                # Get input actions from the viewer and handle them appropriately
                if not args.headless:
                    # # Step rendering
                    gym.step_graphics(sim)
                    gym.draw_viewer(viewer, sim, False)

            # import numpy as np
            # rewards = np.load(
            #     "/home/fuda/PycharmProjects/multi-skill-learning/results/SIM/PPO/spider/gait/spider0/eps_rewards.npy")
            if len(fitnesses) == 300:
                plt.plot(fitnesses)
                plt.plot(np.convolve(fitnesses, np.array([1, 1, 1, 1, 1, 0, 0, 0, 0]) / 5, "same"), '--')
                # plt.show()
                plt.savefig(f"{learner_dir}/rewards.pdf")
                plt.close()

                print(f"Finished learning {skill} on {robot_name}: {rep} {len(fitnesses)} {t}")
                model_path = f"{learner_dir}/model.cleanrl_model"
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")
                np.save(f'{learner_dir}/eps_rewards.npy', np.array(fitnesses))
                writer.close()
            else:
                import shutil
                shutil.rmtree(learner_dir)
            rep += 1
    print(f"destroying simulation: {(time.time() - start_time) / 3600}")
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
