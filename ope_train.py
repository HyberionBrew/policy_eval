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
parser.add_argument('--target_policy', type=str, default="FTG", help="target policy name")
parser.add_argument('--map', type=str, default="Infsaal2", help="map name")
parser.add_argument('--algo', type=str, default="fqe", help="algo name")
parser.add_argument('--experiment_directory', type=str, default="runs", help="experiment directory")
parser.add_argument('--trajectory_length', type=int, default=250, help="trajectory length")
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--eval_interval', type=int, default=10_000, help="eval interval")
parser.add_argument('--reward_config', type=str, default='reward_raceline.json', help='reward config')
parser.add_argument('--update_steps', type=int, default=100_000, help='update steps')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight decay')
parser.add_argument('--load_path', type=str, default=None, help='load path')

args = parser.parse_args()

def get_infinite_iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

def create_save_dir(args):
    save_directory = os.path.join(args.experiment_directory)
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    # now the algo directory
    save_directory = os.path.join(save_directory, f"{args.algo}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    # path
    save_directory = os.path.join(save_directory, f"{args.dataset}_{args.reward_config[:-5]}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)

    #now the max_timesteps directory
    save_directory = os.path.join(save_directory, f"{args.trajectory_length}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    # now the target policy directory
    save_directory = os.path.join(save_directory, f"{args.target_policy}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    return save_directory

def main(args):
    evaluation_agents = ["StochasticContinousFTGAgent_0.5_1_0.2_0.15_0.15_5.0_0.6_0.9",
                         "StochasticContinousFTGAgent_0.75_0_0.2_0.15_0.15_5.0_0.3",
                         "pure_pursuit_1.0_1.2_raceline",
                         "pure_pursuit_0.85_1.0_raceline_og",
                         "pure_pursuit_0.6_0.5_raceline_og_3",
   #                      "StochasticContinousFTGAgent_0.45_1_0.2_0.15_0.15_5.0_0.6_0.9",
                         "StochasticContinousFTGAgent_1.0_7_0.2_0.15_0.15_5.0_0.3",
                         ]
    # setup tensorboard writer
    print("Start OPE")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_path = create_save_dir(args)
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
        include_timesteps_in_obs = True,
        set_terminals=True,
        delta_factor=1.0,
        reward_config=args.reward_config,
        **dict(name=args.dataset,
            config = dict(map=args.map, num_agents=1,
            params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human")
    )

    env = F110Env
    behavior_dataset = F110Dataset(
        env,
        normalize_states=True,
        normalize_rewards=True,
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
        only_agents= args.target_policy,)

    dataloader = DataLoader(behavior_dataset, batch_size=256, shuffle=True)
    inf_dataloader = get_infinite_iterator(dataloader)
    data_iter = iter(inf_dataloader)

    ##### Get the agent loaded in #####
    
    actor = Agent().load(f"/home/fabian/msc/f110_dope/ws_release/config_1501/config/agent_configs/{args.target_policy}.json") # have to tidy this up

    subsample_laser = 20 


    """
    @brief input shape is (batch_size, obs_dim), state needs to be normalized!!
    """
    def get_target_actions(states, scans= None, action_timesteps=None, batch_size=5000,keys=None):
        num_batches = int(np.ceil(len(states) / batch_size))
        actions_list = []
        # batching, s.t. we dont run OOM
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(states))
            batch_states = states[start_idx:end_idx].clone()

            # unnormalize from the dope dataset normalization
            # print(batch_states.shape)
            batch_states_unnorm = behavior_dataset.unnormalize_states(batch_states, keys=keys) # this needs batches
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
            model_input_dict = F110Env.unflatten_batch(batch_states_unnorm, keys=keys)
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

    def get_target_logprobs(states,actions,scans=None,action_timesteps=None, batch_size=5000, keys=None):
        num_batches = int(np.ceil(len(states) / batch_size))
        log_probs_list = []
        for i in range(num_batches):
            # print(i)
            # Calculate start and end indices for the current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(states))
            # Extract the current batch of states
            batch_states = states[start_idx:end_idx]
            batch_states_unnorm = behavior_dataset.unnormalize_states(batch_states, keys=keys)
            
            # Extract the current batch of actions
            batch_actions = actions[start_idx:end_idx]

            # get scans
            if scans is not None:
                laser_scan = scans[start_idx:end_idx].cpu().numpy()
            else:
                laser_scan = F110Env.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
                laser_scan = F110Env.normalize_laser_scan(laser_scan)

            # back to dict
            model_input_dict = F110Env.unflatten_batch(batch_states_unnorm, keys=keys)
            # normalize back to model input
            # model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
            # now also append the laser scan
            model_input_dict['lidar_occupancy'] = laser_scan

            # Compute log_probs for the current batch
            # print(model_input_dict["lidar_occupancy"].shape)
            batch_log_probs = actor(
                model_input_dict,
                actions=batch_actions,
                std=None)[2]
            
            # Sum along the last axis if the rank is greater than 1
            # print("len logprobs", print(batch_log_probs.shape))
            
            # Collect the batch_log_probs
            log_probs_list.append(batch_log_probs)
        # Concatenate the collected log_probs from all batches
        log_probs = [torch.from_numpy(log_prob) for log_prob in log_probs_list]
        log_probs = torch.concat(log_probs, axis=0)
        return log_probs.float()
    #print(behavior_dataset.)
    #print(behavior_dataset.actions[:20])
    #print(get_target_actions(behavior_dataset.states[:20], scans=behavior_dataset.scans[:20]))
    # print(behavior_dataset.states[:10])
    #print(F110Env.keys)
    #print(behavior_dataset.unnormalize_states(behavior_dataset.states[:10]))
    #print(behavior_dataset.unnormalize_states(behavior_dataset.states[:10]))
    #exit()
    discount = 0.99
    update_steps = args.update_steps
    min_reward = behavior_dataset.rewards.min()
    max_reward = behavior_dataset.rewards.max()

    if args.algo == "fqe" or args.algo == "dr":
        from policy_eval.q_fitter import QFitter
        model = QFitter(behavior_dataset.states.shape[1],#env.observation_spec().shape[0],
                            behavior_dataset.actions.shape[1], 
                            critic_lr=3e-5, 
                            weight_decay=5e-6,
                            tau=0.001, 
                            discount = 0.99,
                            use_time=False, # already included in obs now
                            timestep_constant = behavior_dataset.timestep_constant,
                            writer=writer)
        # check if model is on cuda
        #if torch.cuda.is_available():
        #    model.cuda()
    if args.algo == "mb":
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
        
        model.load("/home/fabian/msc/f110_dope/ws_release/mb_testing", filename="model_30000.pt")


    if args.algo == "dr":
        from policy_eval.doubly_robust import DR_estimator
        fqe_load = f"/home/fabian/msc/f110_dope/ws_release/runs/fqe/{args.dataset}_{args.reward_config[:-5]}/{args.trajectory_length}/{args.target_policy}/"
        print("Loading from ", fqe_load)
        model.load(fqe_load,
                    i=0)
        dr_model = DR_estimator(model, behavior_dataset, discount)
        
    # now go to training loop
    pbar = tqdm(range(update_steps), mininterval=5.0)
    
    for i in pbar:
    
        (states, scans, actions, next_states, next_scans, rewards, masks, weights,
        log_prob) = next(data_iter)
        ###### Training Update ######
        if args.algo == "fqe":
            next_actions = get_target_actions(next_states, scans=next_scans)
            max_r = (max_reward*max(250,50)) # the length of the dataset times the maximum reward we can find in it
            # clip actions between 0.3 and -0.3
            #actions = torch.clamp(actions, -0.3, 0.3)
            #next_actions = torch.clamp(next_actions, -0.3, 0.3)

            
            model.update(states, actions, next_states, 
                            next_actions, rewards, masks,
                            weights, discount, 
                            min_reward = min_reward, 
                            max_reward= max_r, 
                            timesteps=None)
        if args.algo == "mb":
            # trained in train_mb_torch.py
            pass
            # model.update(states, actions, next_states, rewards, masks, weights)
        elif args.algo == "dr":
            print("No training required for DR")
            pass
        ###### Evaluation ######


        if i % args.eval_interval == 0:
            if args.algo == "fqe":
                pred_returns, std = model.estimate_returns(behavior_dataset.initial_states,
                                        behavior_dataset.initial_weights,
                                        get_target_actions,
                                        torch.zeros(behavior_dataset.initial_states.shape[0],1))
                pred_returns = behavior_dataset.unnormalize_rewards(pred_returns)
                std = behavior_dataset.unnormalize_rewards(std)

                pred_returns *= (1-discount)
                std *= (1-discount)
                # log it to tensorboard
                model.save(save_path, i=0)

            if args.algo == "dr":
                pred_return, pred_std = dr_model.estimate_returns(get_target_actions, get_target_logprobs, algo="fqe")
                pred_returns = behavior_dataset.unnormalize_rewards(pred_return)
                std = behavior_dataset.unnormalize_rewards(pred_std)
                pred_returns *= (1-discount)
                std *= (1-discount)
            
            if args.algo == "mb":
                from policy_eval.model_based import mb_keys
                # The model is only trained on some of the observations
                filtered_initial = F110Env.get_specific_obs(evaluation_dataset.initial_states, mb_keys)
                # print(filtered_initial.shape)
                pred_returns, std = model.estimate_returns(filtered_initial,
                                        get_target_actions, plot=True)
                pred_returns = behavior_dataset.unnormalize_rewards(pred_returns)
                std = behavior_dataset.unnormalize_rewards(std)

                pred_returns *= (1-discount)
                std *= (1-discount)
                # log it to tensorboard
                # model.save(save_path, i=0)

            pbar.set_description(f"Pred Returns: {pred_returns:.4f}, Std: {std:.4f}")
            writer.add_scalar(f"eval/mean_{args.algo}", pred_returns, global_step=i)
            writer.add_scalar(f"eval/std_{args.algo}", std, global_step=i)

        # early exit if dr
        if args.algo == "dr" or args.algo=="mb":
            break


if __name__ == "__main__":
    main(args)