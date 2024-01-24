from f110_agents.agent import Agent
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt


def plot_pdf_and_logprob(dist, action, label):
    # Generate a range of values around the mean of the distribution
    x_values = np.linspace(dist.mean() - 3*dist.std(), dist.mean() + 3*dist.std(), 1000)
    
    # Calculate the PDF values
    pdf_values = dist.pdf(x_values)
    
    # Calculate the log probability of the action
    log_prob = dist.logpdf(action)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, pdf_values, label=f'PDF of {label}')
    plt.fill_between(x_values, pdf_values, alpha=0.2)
    
    # Plot a vertical line for the action and annotate the log probability
    plt.axvline(action, color='red', linestyle='--', label=f'Action (log prob = {log_prob:.3f})')
    plt.annotate(f'log prob = {log_prob:.3f}', xy=(action, dist.pdf(action)),
                 xytext=(action, dist.pdf(action) + 0.05 * np.max(pdf_values)),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center')
    
    # Additional plot settings
    plt.title(f'Probability Density Function and Log Probability for {label}')
    plt.xlabel(label)
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def log_prob_clipped_norm(actions, fitted_distribution, clipping ):
    """
    Calculate the log probabilities of actions based on a clipped normal distribution.

    Args:
    actions (np.array): Array of action values.
    fitted_distribution (dict): Dictionary containing the fitted distribution and point masses.

    Returns:
    np.array: Array of log probabilities for each action.
    """
    trunc_dist = fitted_distribution["trunc_dist"]
    point_mass_lower = fitted_distribution["point_mass_lower"]
    point_mass_upper = fitted_distribution["point_mass_upper"]

    lower_bound, upper_bound = trunc_dist.ppf(0.001), trunc_dist.ppf(0.999)
    # Ensure bounds are valid
    trunc_dist_not_defined = False
    
    lower_bound, upper_bound = clipping
    try:
        if lower_bound >= upper_bound:
            raise ValueError
        if np.isnan(lower_bound) or np.isnan(upper_bound):
            raise ValueError
    except:
        trunc_dist_not_defined = True
        #print(trunc_dist.a, trunc_dist.b)
        lower_bound, upper_bound = clipping
    
    log_probs = np.zeros_like(actions)

    mask_inside = (actions > lower_bound) & (actions < upper_bound)
    log_probs[mask_inside] = trunc_dist.logpdf(actions[mask_inside])
    # any masks inside that are nan set to -24
    log_probs[np.isnan(log_probs)] = np.log(1e-5) # in case there is no valid distribution defined
    # At lower bound - combine log PDF and log point mass
    mask_lower = actions <= lower_bound
    combined_prob_lower =  point_mass_lower if trunc_dist_not_defined or np.isnan(trunc_dist.pdf(lower_bound)) else trunc_dist.pdf(lower_bound) + point_mass_lower 
    if combined_prob_lower == 0:
        log_probs[mask_lower] = np.log(1e-5)
    else:
        
        log_probs[mask_lower] = np.log(combined_prob_lower)

    # At upper bound - combine log PDF and log point mass
    mask_upper = actions >= upper_bound
    #print(mask_upper)
    #print("mask upper", mask_upper)
    combined_prob_upper = point_mass_upper if trunc_dist_not_defined or np.isnan(trunc_dist.pdf(upper_bound)) else trunc_dist.pdf(upper_bound) + point_mass_upper
    #print("HEÖLLLLOOO")
    #print("combined_prob_upper", combined_prob_upper)
    if combined_prob_upper == 0:
        log_probs[mask_upper] = np.log(1e-5)
    else:
        log_probs[mask_upper] = np.log(combined_prob_upper)
    #print("combined_prob_upper", combined_prob_upper)
    
    assert (not np.isnan(log_probs)) # .all()
    return log_probs


def plot_clipped_norm_with_data(action_data, fitted_distribution, action_og, clipped, title="steering"):
    """
    Plot a histogram of action data with fitted clipped normal distribution and point masses.

    Args:
    action_data (np.array): Array of action data.
    fitted_distribution (dict): Dictionary containing the fitted distribution and point masses.
    """
    trunc_dist = fitted_distribution["trunc_dist"]
    point_mass_lower = fitted_distribution["point_mass_lower"]
    point_mass_upper = fitted_distribution["point_mass_upper"]

    # Generate values for the fitted normal distribution
    x_values = np.linspace(trunc_dist.ppf(0.001), trunc_dist.ppf(0.999), 1000)
    pdf_values = trunc_dist.pdf(x_values)
    # Plot histogram of the action data
    plt.hist(action_data, bins=30, density=True, alpha=0.5, color='blue', label='Action Data')

    # Plot the fitted truncated normal distribution
    plt.plot(x_values, pdf_values, 'k--', label='Fitted Truncated Normal')

    # Indicate point masses at clipping points
    #lower_bound, upper_bound = trunc_dist.ppf(0.001), trunc_dist.ppf(0.999)

    #if not np.isnan(lower_bound) and not np.isnan(upper_bound):
    #    pass
    #else:
        
    lower_bound, upper_bound = clipped
    #print(lower_bound, upper_bound)
    #plt.axvline(lower_bound, color='red', linestyle='--', lw=2, label=f'Point Mass Lower ({point_mass_lower:.2f})')
    #plt.axvline(upper_bound, color='green', linestyle='--', lw=2, label=f'Point Mass Upper ({point_mass_upper:.2f})')

    plt.axvline(lower_bound, color='red', linestyle='--', lw=2, label=f'Point Mass Lower ({point_mass_lower:.2f})')
    plt.axvline(upper_bound, color='green', linestyle='--', lw=2, label=f'Point Mass Upper ({point_mass_upper:.2f})')
    plt.axvline(action_og, color='black', linestyle='dashed', linewidth=2)
    plt.title(f'{title} Data with Fitted Clipped Normal Distribution')
    # plt.xlabel('Action Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def clipped_norm(action_data, lower_bound, upper_bound):
    """
    Fit a truncated normal distribution to an array of action data and calculate point masses at clipping points.

    Args:
    action_data (np.array): Array of action data.
    lower_bound (float): Lower bound for clipping.
    upper_bound (float): Upper bound for clipping.

    Returns:
    dict: Parameters of the fitted distribution and point masses at clipping points.
    """
    # Data within the bounds
    bounded_data = action_data[(action_data > lower_bound) & (action_data < upper_bound)]

    # Estimate mean and standard deviation of the bounded data
    mean, std_dev = norm.fit(bounded_data)

    # Normalize the bounds
    a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev

    # Calculate the point masses at the clipping points
    point_mass_lower = np.sum(action_data <= lower_bound)
    point_mass_upper = np.sum(action_data >= upper_bound)

    # Create the truncated normal distribution
    trunc_dist = truncnorm(a, b, loc=mean, scale=std_dev)

    return {
        "trunc_dist": trunc_dist,
        "point_mass_lower": point_mass_lower,
        "point_mass_upper": point_mass_upper
    }

import os
import pickle

def generate_or_load_noise(filename, num_points, pose_error_std, theta_error_std):
    if os.path.exists(filename):
        # Load the noise from the pickle file
        #print("loading noise from file")
        with open(filename, 'rb') as f:
            x_noise, y_noise, theta_noise = pickle.load(f)
    else:
        # Generate the noise
        x_noise = np.random.normal(0, pose_error_std, num_points)
        y_noise = np.random.normal(0, pose_error_std, num_points)
        theta_noise = np.random.normal(0, theta_error_std, num_points)

        # Save the noise to a pickle file for future use
        with open(filename, 'wb') as f:
            pickle.dump((x_noise, y_noise, theta_noise), f)

    return x_noise, y_noise, theta_noise

class IW:
    def __init__(self, policy,env, gamma=0.99, plot=False):
        self.policy = policy
        self.gamma = gamma
        self.pose_error_std= 0.1
        self.theta_error_std = 0.1
        self.env = env
        self.num_samples_per_point = 35
        self.min_std_deviation = 0.002
        self.plot = plot
    def __call__(self, trajectory):
        """
        trajectory: list of (state, action, reward) tuples
        """
    def get_log_probs(self, policy, states, actions, timeouts, pose_timesteps, action_timesteps):
        """
        trajectory: list of (state, action, reward) tuples
        """
        # need to calculate this
        #log_probs = []
        truncations = timeouts
        ends = np.where(truncations == 1)[0] + 1
        starts = np.where(np.roll(truncations,1) == 1)[0]
        xs = states[:,0]
        ys = states[:,1]
        thetas = states[:,2]
        unflattened = self.env.unflatten_batch(states)
        # print(unflattened.keys())
        prev_act_speed = unflattened["previous_action_speed"]
        prev_act_steer = unflattened["previous_action_steer"]
        log_probs_ = np.zeros((len(xs), 1, 2))
        for start, end in zip(starts, ends):
            # print(start, end)
            poses_x = xs[start:end]
            poses_y = ys[start:end]
            theta = thetas[start:end]
            _prev_act_steer = prev_act_steer[start:end]
            _prev_act_speed = prev_act_speed[start:end]
            timestep_zero = pose_timesteps[start]
            _pose_timesteps = pose_timesteps[start:end] - timestep_zero
            _action_timesteps = action_timesteps[start:end] - timestep_zero

            # create spline 
            spline_x = UnivariateSpline(_pose_timesteps, poses_x, s=0.05)
            spline_y = UnivariateSpline(_pose_timesteps, poses_y, s=0.05)
            unwrapped_thetas = np.unwrap(theta)
            spline_theta = UnivariateSpline(_pose_timesteps, unwrapped_thetas, s=0.05)
            num_samples_per_point = 35
            for i in range(len(_action_timesteps)):
                start_sample = _action_timesteps[i]
                end_sample = _action_timesteps[i-1] if i > 0 else 0.0
                # sample with linspace the points in between
                sampled = np.linspace(start_sample, end_sample, 10, endpoint=True)
                # get from the spline the x and y values
                x_starts = spline_x(sampled)
                y_starts = spline_y(sampled)
                theta_starts = spline_theta(sampled)
                theta_starts = np.arctan2(np.sin(theta_starts), np.cos(theta_starts))
                # Initialize arrays to store the point clouds
                x_cloud = np.zeros((x_starts.size, self.num_samples_per_point))
                y_cloud = np.zeros((y_starts.size, self.num_samples_per_point))
                theta_cloud = np.zeros((theta_starts.size, self.num_samples_per_point))

                filename = 'noise_data.pkl'
                num_samples = x_starts.size * num_samples_per_point  # Total number of noise samples needed
                x_noise, y_noise, theta_noise = generate_or_load_noise(filename, num_samples, self.pose_error_std, self.theta_error_std)

                # Generate point clouds
                for j in range(x_starts.size):
                    x_cloud[j, :] = x_starts[j] + x_noise[j * num_samples_per_point:(j + 1) * num_samples_per_point]
                    y_cloud[j, :] = y_starts[j] + y_noise[j * num_samples_per_point:(j + 1) * num_samples_per_point]
                    theta_cloud[j, :] = np.arctan2(np.sin(theta_starts[j] + theta_noise[j * num_samples_per_point:(j + 1) * num_samples_per_point]), 
                                                    np.cos(theta_starts[j] + theta_noise[j * num_samples_per_point:(j + 1) * num_samples_per_point]))

                # apply 50 noise points to each start 
                x_cloud_flat = x_cloud.flatten()
                y_cloud_flat = y_cloud.flatten()
                theta_cloud_flat = theta_cloud.flatten()
                # Calculate the vector components for each point
                u = np.cos(theta_cloud_flat)  # x component of the vector
                v = np.sin(theta_cloud_flat)  # y component of the vector
                
                # Plot the point clouds with quiver
                """
                plt.figure(figsize=(10, 6))
                plt.quiver(x_cloud_flat, y_cloud_flat, u, v, color='red', scale=50, headwidth=4, headlength=6, headaxislength=5, alpha=0.6)
                # plot the starts
                plt.quiver(x_starts, y_starts, np.cos(theta_starts), np.sin(theta_starts), color='blue', scale=50, headwidth=4, headlength=6, headaxislength=5, alpha=0.6)
                # plot the actual pose
                plt.quiver(poses_x[i], poses_y[i], np.cos(theta[i]), np.sin(theta[i]), color='green', scale=50, headwidth=4, headlength=6, headaxislength=5, alpha=0.6)
                # plot the start and end according to interpolation
                plt.quiver(spline_x(start_sample), spline_y(start_sample), np.cos(spline_theta(start_sample)), np.sin(spline_theta(start_sample)), color='black', scale=50, headwidth=4, headlength=6, headaxislength=5, alpha=0.6)
                plt.quiver(spline_x(end_sample), spline_y(end_sample), np.cos(spline_theta(end_sample)), np.sin(spline_theta(end_sample)), color='black', scale=50, headwidth=4, headlength=6, headaxislength=5, alpha=0.6)
                plt.title('Point Clouds with Theta Direction for Each Point')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True)
                plt.show()
                """
                # now lets calculate the log_prob for each of the points in the point cloud
                obs_batch = dict()
                obs_batch["poses_x"] = x_cloud_flat
                obs_batch["poses_y"] = y_cloud_flat
                obs_batch["poses_theta"] = theta_cloud_flat
                # duplicate the prev actions to have size of the batch
                obs_batch["previous_action_steer"] = np.ones_like(x_cloud_flat) * _prev_act_steer[i] 
                #_prev_act_steer[i]
                obs_batch["previous_action_speed"] = np.ones_like(x_cloud_flat) * _prev_act_speed[i]
                # apply random noise to the prev actions_speed
                # obs_batch["previous_action_speed"] += np.random.normal(0, previous_action_std, obs_batch["previous_action_speed"].shape)
                # print(model_name)
                #print(obs_batch)
                _, action , _ = actor(obs_batch)
                # compute the action we would have taken with the original observation
                obs_batch = dict()
                obs_batch["poses_x"] =  np.array([poses_x[i]])
                obs_batch["poses_y"] = np.array( [poses_y[i]])
                obs_batch["poses_theta"] = np.array([theta[i]])
                # duplicate the prev actions to have size of the batch
                obs_batch["previous_action_steer"] =  np.array([_prev_act_steer[i]])
                #_prev_act_steer[i]
                obs_batch["previous_action_speed"] = np.array([_prev_act_speed[i]])
                #print(obs_batch)
                #_, action_og, _ = actor(obs_batch)
                action_og = actions[i]
                #rint(action_og)
                # plot the action distirbution
                # plt.subplot(1, 2, 1)
                
                action_steering = action[:,0]
                action_speed = action[:,1]
                """
                plt.hist(action_steering, bins='auto', density=True, alpha=0.6, color='green')
                plt.axvline(action_og[i,0], color='red', linestyle='dashed', linewidth=2)
                plt.title('Action Steering Histogram')
                plt.xlabel('Action Steering')
                plt.ylabel('Density')

                # Plot histogram and fitted normal distribution for action_speed
                plt.subplot(1, 2, 2)
                plt.hist(action_speed, bins='auto', density=True, alpha=0.6, color='blue')
                plt.axvline(action_og[i,1], color='red', linestyle='dashed', linewidth=2)
                plt.title('Action Speed with Fitted Normal Distribution')
                plt.xlabel('Action Speed')
                plt.ylabel('Density')
                plt.show()
                """
                #plt.show()
                # now we need to fit propability distributions to the actions
                steering_mean, steering_std = np.mean(action_steering), np.maximum(np.std(action_steering), self.min_std_deviation)
                speed_mean, speed_std = np.mean(action_speed), np.maximum(np.std(action_speed), self.min_std_deviation)

                steering_dist = norm(loc=steering_mean, scale=steering_std)
                speed_dist = norm(loc=speed_mean, scale=speed_std)

                steering_log_prob = steering_dist.pdf(action_og[0])
                speed_log_prob = speed_dist.pdf(action_og[1])
            
                if self.plot:
                    plot_pdf_and_logprob(steering_dist, action_og[0], 'Steering')
                    plot_pdf_and_logprob(speed_dist, action_og[1], 'Speed')

                log_probs_[i + start,0,0] = steering_log_prob
                log_probs_[i +start,0,1] = speed_log_prob
        return log_probs_

    
if __name__ == "__main__":
    import f110_gym
    import f110_orl_dataset
    import gymnasium as gym
    F110Env = gym.make("f110-real-v1",
    # only terminals are available as of right now 
        encode_cyclic=False,
        flatten_obs=True,
        timesteps_to_include=(0,250),
        use_delta_actions=True,
        include_timesteps_in_obs = True,
        set_terminals=True,
        delta_factor=1.0,
        reward_config="reward_progress.json",
        **dict(name="f110-real-v1",
            config = dict(map="Infsaal2", num_agents=1,
            params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human")
    )
    d4rl_dataset = F110Env.get_dataset(
        only_agents=[ 'pure_pursuit_0.7_1.0_raceline_og','pure_pursuit2_0.8_1.2_raceline_og_3_0.6', 'pure_pursuit2_0.9_1.0_raceline_0.6_1.2', 'pure_pursuit_0.6_0.5_raceline_og_3', 'pure_pursuit_0.6_1.0_raceline_og_3', 'pure_pursuit_0.75_0.9_raceline_og', 'pure_pursuit_0.75_0.9_raceline_og_3', 'pure_pursuit_0.7_1.2_raceline_og_3', 'pure_pursuit_0.85_1.0_raceline_og', 'pure_pursuit_0.8_1.0_raceline', 'pure_pursuit_0.8_1.2_raceline_og_3', 'pure_pursuit_0.9_1.0_raceline', 'pure_pursuit_0.9_1.4_raceline_og_3', 'pure_pursuit_1.0_1.2_raceline'],
    )
    log_probs_behavior = np.clip(d4rl_dataset["log_probs"], -7, 2)
    actor = Agent().load(f"/home/fabian/msc/f110_dope/ws_release/config_1501/config/agent_configs/pure_pursuit_0.7_1.0_raceline_og.json")
    model = IW(actor, F110Env, plot=False)
    log_probs =model.get_log_probs(actor, d4rl_dataset["observations"], 
                        d4rl_dataset["actions"], 
                        timeouts=d4rl_dataset["timeouts"], 
                        pose_timesteps=d4rl_dataset["infos"]["pose_timestamp"], 
                        action_timesteps=d4rl_dataset["infos"]["action_timestamp"])
    #get_log_probs(actor, states, actions)
    # clipp the log probs between -7 and 2
    log_probs = np.clip(log_probs, -7, 2)
    log_diff = log_probs - log_probs_behavior
    plt.plot(log_probs[:,0,0])
    plt.plot(log_probs[:,0,1])
    plt.title("Log probs")
    plt.legend(["steering", "speed"])
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