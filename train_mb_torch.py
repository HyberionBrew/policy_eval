from torch.utils.data import DataLoader
from policy_eval.dataset import F110Dataset # this is a dataset wrapper for OPE


import f110_gym
import f110_orl_dataset
import gymnasium as gym

from f110_orl_dataset.normalize_dataset import Normalize
import numpy as np
import torch
from f110_agents.agent import Agent

from tensorboardX import SummaryWriter
import os

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='OPE evaluation')
parser.add_argument('--dataset', type=str, default="f110-real-v1", help="dataset")
parser.add_argument('--target_policy', type=str, default="StochasticContinousFTGAgent_0.5_1_0.2_0.15_0.15_5.0_0.6_0.9", help="target policy")
parser.add_argument('--map', type=str, default="Infsaal2", help="map name")
parser.add_argument('--experiment_directory', type=str, default="runs", help="experiment directory")
parser.add_argument('--trajectory_length', type=int, default=250, help="trajectory length")
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--eval_interval', type=int, default=10_000, help="eval interval")
parser.add_argument('--reward_config', type=str, default='reward_raceline.json', help='reward config')
parser.add_argument('--update_steps', type=int, default=100_000, help='update steps')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight decay')

args = parser.parse_args()

def get_infinite_iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

def main(args):
    evaluation_agents = ["StochasticContinousFTGAgent_0.5_1_0.2_0.15_0.15_5.0_0.6_0.9",
                         "StochasticContinousFTGAgent_0.75_0_0.2_0.15_0.15_5.0_0.3",
                         "pure_pursuit_1.0_1.2_raceline",
                         "pure_pursuit_0.85_1.0_raceline_og",
                         "pure_pursuit_0.6_0.5_raceline_og_3",
                         "StochasticContinousFTGAgent_0.45_1_0.2_0.15_0.15_5.0_0.6_0.9",
                         "StochasticContinousFTGAgent_1.0_7_0.2_0.15_0.15_5.0_0.3",
                         ]
    # setup tensorboard writer
    print("Start OPE")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_path = "mb_testing"
    import datetime
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d-%H-%M-%S")
    file_name = str(args.seed) + "_" + time
    writer = SummaryWriter(log_dir= os.path.join(save_path, file_name))
    print("Logging to ", os.path.join(save_path, file_name))
    subsample_laser = 20

    # load the dataset
    F110Env = gym.make(args.dataset,
    # only terminals are available as of right now 
        encode_cyclic=True,
        flatten_obs=True,
        timesteps_to_include=(0,args.trajectory_length),
        use_delta_actions=True,
        include_timesteps_in_obs = False,
        set_terminals=True,
        delta_factor=1.0,
        reward_config=args.reward_config,
        include_pose_time_diff=False,
        include_action_pose_time_diff = False,
        include_time_obs = False,
        include_progress=False,

        **dict(name=args.dataset,
            config = dict(map=args.map, num_agents=1,
            params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human")
    )

    env = F110Env
    behavior_dataset = F110Dataset(
        env,
        normalize_states=True,
        normalize_rewards=False,
        remove_agents= evaluation_agents,
        #only_agents=['pure_pursuit_2.0_0.15'], #["StochasticContinousFTGAgent_0.15_5_0.2_0.15_2.0"],
        # = ['StochasticContinousFTGAgent_0.5_5_0.2_0.3_2.0',],
        #                  'StochasticContinousFTGAgent_0.5_5_0.2_0.3_2.0',
        #                  'StochasticContinousFTGAgent_5.0_2_0.2_0.3_2.0',
        #                  'StochasticContinousFTGAgent_5.0_5_0.2_0.3_2.0'],#,'StochasticContinousFTGAgent_3.0_5_0.2_0.3_2.0'], #'progress_weight', 'raceline_delta_weight', 'min_action_weight'],#['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
        
    )
    evaluation_dataset = F110Dataset(
        env,
        normalize_states=True,
        normalize_rewards=False,
        only_agents=evaluation_agents,
        state_mean=behavior_dataset.state_mean,
        state_std=behavior_dataset.state_std,
        #reward_mean=behavior_dataset.reward_mean,
        #reward_std=behavior_dataset.reward_std,
    )

    dataloader = DataLoader(behavior_dataset, batch_size=256, shuffle=True)
    inf_dataloader = get_infinite_iterator(dataloader)
    data_iter = iter(inf_dataloader)

    ##### Get the agent loaded in #####
    subsample_laser = 20 
    
    ##### Get the agent loaded in #####
    
    actor = Agent().load(f"/home/fabian/msc/f110_dope/ws_release/config_1501/config/agent_configs/{args.target_policy}.json") # have to tidy this up

    subsample_laser = 20 


    """
    @brief input shape is (batch_size, obs_dim), state needs to be normalized!!
    """
    def get_target_actions(states, scans= None, action_timesteps=None, batch_size=5000):
        num_batches = int(np.ceil(len(states) / batch_size))
        actions_list = []
        # batching, s.t. we dont run OOM
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(states))
            batch_states = states[start_idx:end_idx].clone()

            # unnormalize from the dope dataset normalization
            batch_states_unnorm = behavior_dataset.unnormalize_states(batch_states) # this needs batches
            del batch_states
            batch_states_unnorm = batch_states_unnorm.cpu().numpy()

            # get scans
            if scans is not None:
                laser_scan = scans[start_idx:end_idx].cpu().numpy()
            else:
                laser_scan = F110Env.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
                #print("Scan 1")
                #print(laser_scan)
                laser_scan = F110Env.normalize_laser_scan(laser_scan)
            #print("Scan 2")
            #print(laser_scan)
            # back to dict
            #print(batch_states_unnorm.shape)
            model_input_dict = F110Env.unflatten_batch(batch_states_unnorm)
            # normalize back to model input
            # model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
            # now also append the laser scan
            # print(model_input_dict)
            model_input_dict['lidar_occupancy'] = laser_scan
            #print("model input dict")
            #print("after unflattening")
            #print(model_input_dict)
            batch_actions = actor(
            model_input_dict,
            std=None)[1]
            #print(batch_actions)
            
            actions_list.append(batch_actions)
        # tf.concat(actions_list, axis=0)
        # with torch
        # convert to torch tensor
        actions_list = [torch.from_numpy(action) for action in actions_list]
        actions = torch.concat(actions_list, axis=0)
        # print(actions)
        return actions.float()


    discount = 0.99
    update_steps = args.update_steps
    min_reward = behavior_dataset.rewards.min()
    max_reward = behavior_dataset.rewards.max()
    from policy_eval.model_based import ModelBased
    model = ModelBased(F110Env, behavior_dataset.states.shape[1],
                    behavior_dataset.actions.shape[1], 
                    hidden_size = [256,256,256,256],
                    dt=1/20,
                    min_state=behavior_dataset.states.min(axis=0)[0],
                    max_state=behavior_dataset.states.max(axis=0)[0],
                    dataset = behavior_dataset,
                    fn_normalize=behavior_dataset.normalize_states,
                    fn_unnormalize=behavior_dataset.unnormalize_states,
                    obs_keys=behavior_dataset.obs_keys,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        target_reward="reward_progress.json",
                        logger=writer,)

        
    # now go to training loop
    pbar = tqdm(range(update_steps), mininterval=5.0)
    
    for i in pbar:
    
        (states, scans, actions, next_states, next_scans, rewards, masks, weights,
        log_prob) = next(data_iter)
        #print(behavior_dataset.obs_keys)
        loss = model.update(states, actions, next_states, rewards, masks)
        writer.add_scalar(f"train/loss_mb", loss, global_step=i)
        ###### Evaluation ######
        if i % args.eval_interval == 0:
          # evaluate the mse for each agent in the eval dataset
            all_mse = []
            for agent in np.unique(evaluation_dataset.model_names):
                actor_eval = Agent().load(f"/home/fabian/msc/f110_dope/ws_release/config_1501/config/agent_configs/{agent}.json") 
                def get_target_action_eval(states, scans= None, action_timesteps=None, batch_size=5000):
                    num_batches = int(np.ceil(len(states) / batch_size))
                    actions_list = []
                    # batching, s.t. we dont run OOM
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, len(states))
                        batch_states = states[start_idx:end_idx].clone()

                        # unnormalize from the dope dataset normalization
                        batch_states_unnorm = behavior_dataset.unnormalize_states(batch_states) # this needs batches
                        del batch_states
                        batch_states_unnorm = batch_states_unnorm.cpu().numpy()

                        # get scans
                        if scans is not None:
                            laser_scan = scans[start_idx:end_idx].cpu().numpy()
                        else:
                            laser_scan = F110Env.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
                            #print("Scan 1")
                            #print(laser_scan)
                            laser_scan = F110Env.normalize_laser_scan(laser_scan)
                        #print("Scan 2")
                        #print(laser_scan)
                        # back to dict
                        #print(batch_states_unnorm.shape)
                        model_input_dict = F110Env.unflatten_batch(batch_states_unnorm)
                        # normalize back to model input
                        # model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
                        # now also append the laser scan
                        # print(model_input_dict)
                        model_input_dict['lidar_occupancy'] = laser_scan
                        #print("model input dict")
                        #print("after unflattening")
                        #print(model_input_dict)
                        batch_actions = actor_eval(
                        model_input_dict,
                        std=None)[1]
                        #print(batch_actions)
                        
                        actions_list.append(batch_actions)
                    # tf.concat(actions_list, axis=0)
                    # with torch
                    # convert to torch tensor
                    actions_list = [torch.from_numpy(action) for action in actions_list]
                    actions = torch.concat(actions_list, axis=0)
                    # print(actions)
                    return actions.float()

                #print(evaluation_dataset.initial_states.shape)
                pred_returns, std = model.estimate_returns(evaluation_dataset.initial_states, get_target_action=get_target_action_eval)
                print(agent, pred_returns, std)
                N = len(evaluation_dataset.initial_states)
                TRAJECTORY_LENGTH = 250

                # Initialize the arrays for states and actions
                print(evaluation_dataset.mask_inital)
                num_trajectories = np.sum(evaluation_dataset.mask_inital.numpy())
                states_padded = np.zeros((num_trajectories, TRAJECTORY_LENGTH, evaluation_dataset.states.shape[1]))
                actions_padded = np.zeros((num_trajectories, TRAJECTORY_LENGTH, 2))

                # Identify start and end indices for each trajectory
                starts = np.where(evaluation_dataset.mask_inital.numpy())[0]
                ends = np.where(evaluation_dataset.finished)[0]

                # Fill in the trajectories
                trajectory_idx = 0
                for start, end in zip(starts, ends):
                    trajectory_length = min(end - start, TRAJECTORY_LENGTH)

                    # Fill states and actions for each trajectory
                    states_padded[trajectory_idx, :trajectory_length, :] = evaluation_dataset.states[start:end, :TRAJECTORY_LENGTH]
                    actions_padded[trajectory_idx, :trajectory_length, :] = evaluation_dataset.actions[start:end, :2]

                    trajectory_idx += 1
                mse = model.estimate_mse_pose(torch.from_numpy(states_padded.astype(np.float32)), 
                                             # torch.from_numpy(actions_padded.astype(np.float32)), 
                                              get_target_action_eval)
                all_mse.append(mse)
                print("mse:", mse)
                # break
            
        

            pred_returns, std = model.estimate_returns(evaluation_dataset.initial_states, get_target_action=
                            get_target_actions)
            pred_returns = behavior_dataset.unnormalize_rewards(pred_returns)
            std = behavior_dataset.unnormalize_rewards(std)

            pred_returns *= (1-discount)
            std *= (1-discount)
            # log it to tensorboard
            model.save(save_path, filename=f"model_{i}.pt")

            pbar.set_description(f"Pred Returns: {pred_returns:.4f}, Std: {std:.4f}, eval mse: {np.mean(np.array(all_mse)):.4f}")
            writer.add_scalar(f"eval/mean_mb", pred_returns, global_step=i)
            writer.add_scalar(f"eval/std_mb", std, global_step=i)
            writer.add_scalar(f"eval/mse", np.mean(np.array(all_mse)), global_step=i)


if __name__ == "__main__":
    main(args)