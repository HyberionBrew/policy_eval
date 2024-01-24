from f110_agents.agent import Agent
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt


class IW:
    def __init__(self, policy,env, gamma=0.99):
        self.policy = policy
        self.gamma = gamma
        self.pose_error_std= 0.2
        self.theta_error_std = 0.2
        self.env = env
        self.num_samples_per_point = 35

    def __call__(self, trajectory):
        """
        trajectory: list of (state, action, reward) tuples
        """

    
    def get_target_logprobs(self,F110Env, actor, states,actions,scans=None,action_timesteps=None, batch_size=5000):
        """
        Expects unnormalized states
        """
        subsample_laser = 20 
        num_batches = int(np.ceil(len(states) / batch_size))
        log_probs_list = []
        for i in range(num_batches):
            # print(i)
            # Calculate start and end indices for the current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(states))
            # Extract the current batch of states
            batch_states = states[start_idx:end_idx]
            batch_states_unnorm = batch_states # behavior_dataset.unnormalize_states(batch_states)
            
            # Extract the current batch of actions
            batch_actions = actions[start_idx:end_idx]

            # get scans
            if scans is not None:
                laser_scan = scans[start_idx:end_idx] # .cpu().numpy()
            else:
                laser_scan = F110Env.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
                laser_scan = F110Env.normalize_laser_scan(laser_scan)

            # back to dict
            model_input_dict = F110Env.unflatten_batch(batch_states_unnorm)
            # normalize back to model input
            # model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
            # now also append the laser scan
            model_input_dict['lidar_occupancy'] = laser_scan

            # Compute log_probs for the current batch
            # print(model_input_dict["lidar_occupancy"].shape)
            batch_log_probs = actor(
                model_input_dict,
                action=batch_actions,
                std=None)[2]
            #print(batch_log_probs)
            # Sum along the last axis if the rank is greater than 1
            # print("len logprobs", print(batch_log_probs.shape))
            
            # Collect the batch_log_probs
            log_probs_list.append(batch_log_probs)
        # Concatenate the collected log_probs from all batches
        log_probs = [log_prob for log_prob in log_probs_list]
        log_probs = np.concatenate(log_probs, axis=0)
        # to float 32
        log_probs = log_probs.astype(np.float32)
        # add an empty 1 axis
        print(log_probs.shape)
        log_probs = log_probs[:,None,:]

        return log_probs
    
if __name__ == "__main__":
    import f110_gym
    import f110_orl_dataset
    import gymnasium as gym
    F110Env = gym.make("f110-sim-v1",
    # only terminals are available as of right now 
        encode_cyclic=False,
        flatten_obs=True,
        timesteps_to_include=(0,250),
        use_delta_actions=True,
        include_timesteps_in_obs = True,
        set_terminals=True,
        delta_factor=1.0,
        reward_config="reward_progress.json",
        **dict(name="f110-sim-v1",
            config = dict(map="Infsaal2", num_agents=1,
            params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human")
    )
    d4rl_dataset = F110Env.get_dataset(
        zarr_path="/home/fabian/msc/f110_dope/ws_release/iw_dataset2.zarr",
        remove_agents = ["StochasticContinousFTGAgent_0.65_0_0.2_0.1_0.3_5.0_0.1_0.5",'StochasticContinousFTGAgent_0.45_1_0.2_0.1_0.3_5.0_0.6_0.5'],
             # only_agents=[#'StochasticContinousFTGAgent_0.55_0_0.2_0.1_0.3_5.0_0.4_0.5',
              #             'StochasticContinousFTGAgent_0.45_1_0.2_0.1_0.3_5.0_0.6_0.5',
# 'pure_pursuit2_0.6_0.5_raceline_og_3_0.3_0.5', 
 #'pure_pursuit2_0.6_1.0_raceline_og_3_0.3_0.5', 
#'pure_pursuit2_0.75_0.9_raceline_og_0.3_0.5',
 #'pure_pursuit2_0.75_0.9_raceline_og_3_0.3_0.5',
  # 'pure_pursuit2_0.7_1.0_raceline_og_0.3_0.5', 
 #  'pure_pursuit2_0.7_1.2_raceline_og_3_0.3_0.5', 
 #  'pure_pursuit2_0.85_1.0_raceline_og_0.3_0.5', 
 #  'pure_pursuit2_0.8_1.0_raceline_0.3_0.5', 
 #  'pure_pursuit2_0.8_1.2_raceline_og_3_0.3_0.5', 
 #  'pure_pursuit2_0.9_1.0_raceline_0.3_0.5', 
 #  'pure_pursuit2_0.9_1.4_raceline_og_3_0.3_0.5', 
 #  'pure_pursuit2_1.0_1.2_raceline_0.3_0.5',
  # 'StochasticContinousFTGAgent_0.75_0_0.2_0.1_0.3_5.0_0.3_0.5',

   #                        ],#, 'pure_pursuit2_0.8_1.2_raceline_og_3_0.6_0.5', 'pure_pursuit2_0.9_1.0_raceline_0.3_0.5', 'pure_pursuit2_0.9_1.0_raceline_0.6_0.5', 'pure_pursuit2_0.9_1.4_raceline_og_3_0.3_0.5', 'pure_pursuit2_1.0_1.2_raceline_0.3_0.5'],
        #[ ],
        #only_agents=[ "pure_pursuit2_0.7_1.2_raceline_og_3_0.3_0.5",
#'pure_pursuit2_0.8_1.2_raceline_og_3_0.3_0.5', 'pure_pursuit2_0.8_1.2_raceline_og_3_0.6_0.5', 'pure_pursuit2_0.9_1.0_raceline_0.3_0.5', 'pure_pursuit2_0.9_1.0_raceline_0.6_0.5', 'pure_pursuit2_0.9_1.4_raceline_og_3_0.3_0.5', 'pure_pursuit2_1.0_1.2_raceline_0.3_0.5',
#  ],
    )
    gamma = 0.99
    log_probs_behavior = np.clip(d4rl_dataset["log_probs"], -7, 2)
    actor = Agent().load(f"/home/fabian/msc/f110_dope/ws_release/config_1501/config/agent_configs_stoch/StochasticContinousFTGAgent_0.65_0_0.2_0.15_0.15_5.0_0.1.json") #pure_pursuit_1.0_1.2_raceline.json")
    #actor2 = Agent().load(f"/home/fabian/msc/f110_dope/ws_release/config_1501/config/agent_configs_stoch/pure_pursuit_0.75_0.9_raceline_og_3.json")
    model = IW(actor, F110Env)
    end = 7
    #test = model.get_target_logprobs(F110Env, actor1, d4rl_dataset["observations"][5:end], 
    #                    d4rl_dataset["actions"][5:end],
    #                    scans=d4rl_dataset["scans"][5:end])
    #print(test)
    #print(  d4rl_dataset["actions"][5:end])
    #print("-----change-----")
    #test = model.get_target_logprobs(F110Env, actor2, d4rl_dataset["observations"][5:end], 
    #                d4rl_dataset["actions"][5:end],
    #                scans=d4rl_dataset["scans"][5:end])
    #print(test)
    #exit()
    log_probs =model.get_target_logprobs(F110Env, actor, d4rl_dataset["observations"], 
                        d4rl_dataset["actions"],
                        scans=d4rl_dataset["scans"])
    #get_log_probs(actor, states, actions)
    # clipp the log probs between -7 and 2
    log_probs = np.clip(log_probs, -7, 2)
    log_diff =  log_probs - log_probs_behavior
    plt.plot(log_probs[:250,0,0])
    
    plt.plot(log_probs_behavior[:250,0,0])

    plt.title("Log probs")
    plt.legend(["calc", "behavior"])
    plt.show()
    plt.plot(log_probs[:250,0,1])
    plt.plot(log_probs_behavior[:250,0,1])
    plt.legend(["calc", "behavior"])
    plt.title("Log probs")
    plt.show()
    # plot each trajectory the log_diff
    truncated = d4rl_dataset["timeouts"]


    # for each of the contained model plot the appropriate log_probs calculated
    # colormap for each model_name 2
    unique_models = np.unique(d4rl_dataset["model_name"])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_models)))

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plotting
    for i, model_name in enumerate(unique_models):
        relevant_indices = np.where(d4rl_dataset["model_name"] == model_name)[0]
        shifted_truncated = np.roll(truncated[relevant_indices], 1)
        
        # Find start and end indices relative to the relevant_indices
        relative_starts = np.where(shifted_truncated == 1)[0]
        relative_ends = np.where(truncated[relevant_indices] == 1)[0]
        
        # Adjust indices to refer to positions in the original array
        starts = relevant_indices[relative_starts]
        ends = relevant_indices[relative_ends]
        # print(starts)
        first = True
        for start, end in zip(starts, ends):
            if first:
                #print(np.cumsum(log_diff[start:end, 0, 0]))
                ax1.plot(np.cumsum(log_diff[start:end, 0, 0]), color=colors[i], label=f"steering {model_name}")
                ax2.plot(np.cumsum(log_diff[start:end, 0, 1]), color=colors[i], label=f"speed {model_name}")
                first = False
            else:
                
                ax1.plot(np.cumsum(log_diff[start:end, 0, 0]), color=colors[i])
                ax2.plot(np.cumsum(log_diff[start:end, 0, 1]), color=colors[i])

    # Set labels and titles
    ax1.set_title("Steering")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cumulative Steering")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Speed")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Cumulative Speed")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    # find the total number of trajectories
    shifted_truncated = np.roll(truncated, 1)
    all_starts = np.where(shifted_truncated == 1)[0]
    all_ends = np.where(truncated == 1)[0]

    num_trajectories = np.sum(truncated)
    # create log_prob trajectory array
    # calculate the log_probability of each trajectory
    log_probs_trajectory = np.zeros((num_trajectories, 2))
    returns = np.zeros(num_trajectories)
    # for each trajectory
    for i, (start, end) in enumerate(zip(all_starts, all_ends)):
        log_probs_trajectory[i, 0] = np.sum(log_diff[start:end, 0, 0])
        log_probs_trajectory[i, 1] = np.sum(log_diff[start:end, 0, 1])
        # calculated discounted sum of rewards
        returns[i] = np.sum(d4rl_dataset["rewards"][start:end] * np.power(gamma, np.arange(end - start)))
    # sum up the log_probs across the last axis
    log_probs_trajectory = np.sum(log_probs_trajectory, axis=-1)
    probs_trajectory = np.exp(log_probs_trajectory)
    # normalize probs_trajectory to sum up to 1.0
    new_probs = np.zeros_like(probs_trajectory)
    normalized_weights = np.zeros_like(probs_trajectory)

    # Outer loop for each starting position
    for starting_position in range(10):
        # Accumulate the weights for the specific starting position across all batches
        batch_elements = np.arange(starting_position, len(probs_trajectory), 10)
        sum_weights = np.sum(probs_trajectory[batch_elements])

        # Inner loop to normalize weights for each occurrence of the starting position
        for i in batch_elements:
            normalized_weights[i] = probs_trajectory[i] / sum_weights

    print(normalized_weights)
    # find max index in probs_trajectory
    max_index = np.argmax(normalized_weights)
    print("max idx", max_index)
    # find the corresponding trajectory
    print(log_probs_trajectory[max_index])
    print(returns[max_index])
    print(returns)
    # multiply the probs_trajectory with the returns
    weighted_returns = normalized_weights * returns
    # mean the weighted returns
    print(np.sum(weighted_returns, axis=0)/10)
