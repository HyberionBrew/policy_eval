# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of model based policy evaluation."""
import tensorflow as tf
from tensorflow_addons import optimizers as tfa_optimizers
import tensorflow_probability as tfp


import matplotlib.pyplot as plt
import numpy as np

class ForwardModel(tf.keras.Model):
  """A class that implement a forward model."""

  def __init__(self, state_dim, action_dim):
    super(ForwardModel, self).__init__()
    self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256,
            input_dim=state_dim + action_dim,
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(state_dim, kernel_initializer='orthogonal')
    ])

  @tf.function
  def call(self, x, a):
    x_a = tf.concat([x, a], -1)
    # set the first two states in the model to 0
    x_a = tf.concat([tf.zeros_like(x_a[:, :2]), x_a[:, 2:]], -1)
    # set states 9 and 10 to 0
    x_a = tf.concat([x_a[:, :9], tf.zeros_like(x_a[:, 9:11]), x_a[:, 11:]], -1)
    return self.model(x_a) + x


class RewardModel(tf.keras.Model):
  """A class that implement a reward model."""

  def __init__(self, state_dim, action_dim):
    super(RewardModel, self).__init__()
    self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256,
            input_dim=state_dim + action_dim,
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(1, kernel_initializer='orthogonal')
    ])

  @tf.function
  def call(self, x, a):
    x_a = tf.concat([x, a], -1)
    return tf.squeeze(self.model(x_a), 1)

from model_based_2 import GroundTruthReward
standard_config = {
    "collision_penalty": 0.0,
    "progress_weight": 1.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "min_action_weight" : 0.0,
    "min_lidar_ray_weight" : 0.0,
    "inital_velocity": 1.5,
    "normalize": False,
}

class ModelBased(object):
  """A class that learns models and estimated returns via rollouts."""

  def __init__(self, state_dim, action_dim, learning_rate, weight_decay):
    """Creates networks and optimizers for model based policy evaluation.

    Args:
      state_dim: State size.
      action_dim: Action size.
      learning_rate: Critic learning rate.
      weight_decay: Weight decay.
    """
    self.dynamics_net = ForwardModel(state_dim, action_dim)
    self.rewards_net = RewardModel(state_dim, action_dim)
    self.done_net = RewardModel(state_dim, action_dim)

    self.dyn_optimizer = tfa_optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)
    self.reward_optimizer = tfa_optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)
    self.done_optimizer = tfa_optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)
  
  def forward(self, states, actions, min_reward, max_reward, min_state, max_state, clip):
    """
    Forward pass through the model.
    """
    pred_rewards = self.rewards_net(states, actions)
    if clip:
      pred_rewards = tf.clip_by_value(pred_rewards, min_reward,
                                      max_reward)
    
    logits = self.done_net(states, actions)

    states = self.dynamics_net(states, actions)
    
    if clip:
      states = tf.clip_by_value(states, min_state, max_state)
    
    return states, pred_rewards, logits
  

  def evaluate_rollouts(self, dataset, unnormalize_fn, 
                        horizon=100, num_samples=20, discount=1.0):
    reward_model = GroundTruthReward("Infsaal",dataset,20, **standard_config)
    states, actions, rewards, inital_mask = dataset.states, dataset.actions, dataset.rewards, dataset.mask_inital        
    states = tf.stop_gradient(states)
    actions = tf.stop_gradient(actions)
    rewards = tf.stop_gradient(rewards)
    # next_states = tf.stop_gradient(next_states)
    sampled_initial_indices = self.sample_initial_states(inital_mask, min_distance=horizon, num_samples=num_samples)
    sampled_initial_indices = tf.convert_to_tensor(sampled_initial_indices)
    discount_factors = tf.convert_to_tensor([discount**i for i in range(horizon)])

    indices_range = tf.range(horizon)
    # Broadcasting addition to get the indices for slices.
    indices_for_segments = sampled_initial_indices[:, tf.newaxis] + indices_range
    gt_rewards_segments = tf.gather(rewards, indices_for_segments)

    gt_rewards = tf.reduce_sum(gt_rewards_segments * discount_factors, axis=1)

    mean_gt_rewards_tensor = tf.reduce_mean(gt_rewards)

    mean_gt_rewards = unnormalize_fn(mean_gt_rewards_tensor.numpy().item())

    model_rewards = []
    for idx in sampled_initial_indices:
      state = tf.expand_dims(states[idx], axis=0)
      rollout_rewards = []
      reward_model.reset(state[0, :2].numpy(), velocity=1.5)
      for i in range(horizon):
        action = tf.expand_dims(actions[idx + i], axis=0)
        
        next_state = self.dynamics_net(state, action)
        
        # Ensure state and action tensors are converted to numpy arrays for reward_model call.
        pred_reward = reward_model(state.cpu().numpy(), action.cpu().numpy())
        
        # Calculate discounted reward
        discounted_reward = pred_reward * (discount**i)
        
        rollout_rewards.append(discounted_reward)
        
        state = next_state

      model_rewards.append(tf.reduce_sum(rollout_rewards))
    mean_model_rewards = tf.reduce_mean(tf.convert_to_tensor(model_rewards))
    mean_gt_rewards = mean_gt_rewards.numpy()
    tf.summary.scalar('test/mean_gt_rewards', mean_gt_rewards, step=self.dyn_optimizer.iterations)
    tf.summary.scalar('test/mean_model_rewards', mean_model_rewards, step=self.dyn_optimizer.iterations)
    diff = mean_gt_rewards - mean_model_rewards
    tf.summary.scalar('test/model_diff', diff, step=self.dyn_optimizer.iterations)
    return mean_gt_rewards, mean_model_rewards, diff

  def evaluate(self, states, actions, rewards, next_states,
              step,
              name,
              clip= True,
              min_reward=-100,
              max_reward=100,
              min_state=-100,
              max_state=100,):

    """
    Evaluate the model based on the given data.
    """
    states = tf.stop_gradient(states)
    actions = tf.stop_gradient(actions)
    rewards = tf.stop_gradient(rewards)
    next_states = tf.stop_gradient(next_states)

    pred_state, pred_rewards, logits = self.forward(states, actions, min_reward, max_reward, min_state, max_state, clip)
    # compute the loss
    loss = tf.losses.mean_squared_error(next_states, pred_state)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar(f"test/dyn_loss_{name}", loss,
                        step=self.dyn_optimizer.iterations)
    loss_reward = tf.losses.mean_squared_error(rewards, pred_rewards)
    loss_reward = tf.reduce_mean(loss_reward)
    tf.summary.scalar(f"test/reward_loss_{name}", loss,
                        step=self.dyn_optimizer.iterations)

  @tf.function
  def update(self, states, actions,
             next_states, rewards, masks,
             weights):
    """Updates critic parameters.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      next_states: A batch of next states.
      rewards: A batch of rewards.
      masks: A batch of masks.
      weights: A batch of weights.

    Returns:
      Critic loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.dynamics_net.trainable_variables)
      pred_state = self.dynamics_net(states, actions)
      dyn_loss = tf.losses.mean_squared_error(next_states, pred_state)
      dyn_loss = tf.reduce_mean(dyn_loss * weights)

    grads = tape.gradient(dyn_loss, self.dynamics_net.trainable_variables)

    self.dyn_optimizer.apply_gradients(
        zip(grads, self.dynamics_net.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.rewards_net.trainable_variables)
      pred_rewards = self.rewards_net(states, actions)
      reward_loss = tf.losses.mean_squared_error(rewards, pred_rewards)
      reward_loss = tf.reduce_mean(reward_loss * weights)

    grads = tape.gradient(reward_loss, self.rewards_net.trainable_variables)

    self.reward_optimizer.apply_gradients(
        zip(grads, self.rewards_net.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.done_net.trainable_variables)
      pred_dones = self.done_net(states, actions)
      done_loss = tf.keras.losses.binary_crossentropy(
          masks, pred_dones, from_logits=True)
      done_loss = tf.reduce_mean(done_loss * weights)

    grads = tape.gradient(done_loss, self.done_net.trainable_variables)

    self.done_optimizer.apply_gradients(
        zip(grads, self.done_net.trainable_variables))
    
    if self.dyn_optimizer.iterations % 1000 == 0:
      tf.summary.scalar('train/dyn_loss', dyn_loss,
                        step=self.dyn_optimizer.iterations)
      tf.summary.scalar('train/rew_loss', reward_loss,
                        step=self.reward_optimizer.iterations)
      tf.summary.scalar('train/done_loss', done_loss,
                        step=self.done_optimizer.iterations)
  def save(self, path):
    """Saves the model.

    Args:
      path: Path to save the model.
    """
    self.dynamics_net.save_weights(path + '/dynamics_net')
    self.rewards_net.save_weights(path + '/rewards_net')
    self.done_net.save_weights(path + '/done_net')
  def load(self, path):
    """Loads the model.

    Args:
      path: Path to load the model.
    """
    self.dynamics_net.load_weights(path + '/dynamics_net')
    self.rewards_net.load_weights(path + '/rewards_net')
    self.done_net.load_weights(path + '/done_net')
  
  def get_rewards(self, states, 
                  weights, 
                  get_action, 
                  discount,   
                  min_reward,
                       max_reward,
                       min_state,
                       max_state,
                       clip=True,
                       horizon=1000):
    """Compute returns via rollouts.
    """
    returns = 0
    states = states

    masks = tf.ones((states.shape[0],), dtype=tf.float32)
    # do a plot of the trajectory taken by the agent where states[:,0] and states[:,1] are the x and y coordinates of the agent
    for i in range(horizon):
      # print("i", i)
      actions = get_action(states)

      pred_rewards = self.rewards_net(states, actions)
      if clip:
        pred_rewards = tf.clip_by_value(pred_rewards, min_reward,
                                        max_reward)
      logits = self.done_net(states, actions)
      # print(logits.shape)
      mask_dist = tfp.distributions.Bernoulli(logits=logits)
      masks *= tf.cast(mask_dist.sample(), tf.float32)

      returns += (discount**i) * masks * pred_rewards

      states = self.dynamics_net(states, actions)
      if clip:
        states = tf.clip_by_value(states, min_state, max_state)

      return returns
  
  def compute_rewards_for_states(self, states, get_action):
      actions = get_action(states)
      rewards = self.rewards_net(states, actions)
      return rewards.numpy()
  
  def sample_initial_states(self, initial_mask, min_distance=50, num_samples=20):
      # Find all initial state indices
      initial_indices = np.where(initial_mask == 1)[0]

      # Filter out the indices that don't have at least min_distance before the next initial state
      valid_indices = [idx for i, idx in enumerate(initial_indices[:-1]) if initial_indices[i+1] - idx >= min_distance]

      # If the last state is also valid (i.e., it has more than min_distance states until the end of the array)
      if len(initial_mask) - initial_indices[-1] >= min_distance:
          valid_indices.append(initial_indices[-1])

      # If there are fewer or equal valid indices than num_samples, return all valid indices
      if len(valid_indices) <= num_samples:
          return valid_indices
      
      # Determine the step size to take to get num_samples from the list of valid_indices
      step_size = len(valid_indices) // num_samples

      # Select the indices
      sampled_indices = valid_indices[::step_size][:num_samples]
      
      return sampled_indices
  
  def plot_rollouts_fixed(self, states, 
                    actions, 
                    inital_mask,
                    min_state,
                    max_state,
                    clip=True,
                    horizon=1000,
                    num_samples = 20,
                    path="logdir/plts/mb/rollouts_mb.png"):

      # set to no gradients
      states = tf.stop_gradient(states)
      actions = tf.stop_gradient(actions)

      sampled_initial_indices = self.sample_initial_states(inital_mask,min_distance=horizon ,num_samples=num_samples)
      sampled_states = tf.gather(states, sampled_initial_indices)
      
      plt.figure(figsize=(12, 6))
      colors = plt.cm.jet(np.linspace(0, 1, num_samples))
      
      # Plot all states as a grey background
      plt.scatter(states[:, 0], states[:, 1], color='grey', s=5, label='All states', alpha=0.5)

      for idx, start_idx in enumerate(sampled_initial_indices):
          state_trajectory = [states[start_idx][:2]]  # start with the sampled state

          # Plot "x" at the starting state
          plt.scatter(state_trajectory[0][0], state_trajectory[0][1], color=colors[idx], marker='x', s=60, label=f'Start {idx + 1}')
          # Plot the actual states using dashed lines
          actual_trajectory = states[start_idx:start_idx+horizon, :2]
          plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], '--', label=f"Ground Truth {idx + 1}", color=colors[idx])
        

          current_state = np.expand_dims(states[start_idx], 0)  # make it (1, state_dim)
          for i in range(horizon):
              action = np.expand_dims(actions[start_idx + i], 0) 
              
              current_state = self.dynamics_net(current_state, action)
              if clip:
                  current_state = np.clip(current_state, min_state, max_state)

              # Collect x, y for plotting
              state_trajectory.append(current_state[0, :2])

          x_coords, y_coords = zip(*state_trajectory)
          plt.plot(x_coords, y_coords, label=f"Sample {idx + 1}", color=colors[idx])

      # Additional plot formatting can be done here if needed, like setting axis labels, titles, legends, etc.
      # plt.legend(loc='upper left')
      plt.title("Rollouts using Given Actions")
      plt.savefig(path)
      plt.show()


  def plot_rollouts(self, states, 
                      weights, 
                      get_action, 
                      discount,   
                      min_reward,
                      max_reward,
                      min_state,
                      max_state,
                      clip=True,
                      horizon=1000,
                      num_samples = 20,
                      path="logdir/plts/mb/rollouts_mb.png"):

        # sampled_indices = tf.random.shuffle(tf.range(states.shape[0]))[:num_samples]
        # sampled_indices = states[:num_samples]
        sampled_indices = np.linspace(0, states.shape[0]-1, num_samples, dtype=int)
        sampled_states = tf.gather(states, sampled_indices)
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.jet(np.linspace(0, 1, num_samples))  # Get 10 different colors
        # Plot all states as a grey background
        plt.scatter(states[:, 0], states[:, 1], color='grey', s=5, label='All states', alpha=0.5)

        for idx in range(num_samples):
            state_trajectory = [sampled_states[idx][:2].numpy()]  # start with the sampled state
            current_state = sampled_states[idx:idx+1, :]
            masks = tf.ones((1,), dtype=tf.float32)
            returns = 0
            # Plot "x" at the starting state
            plt.scatter(state_trajectory[0][0], state_trajectory[0][1], color=colors[idx], marker='x', s=60, label=f'Start {idx + 1}')
            for i in range(horizon):
                actions = get_action(current_state)

                pred_rewards = self.rewards_net(current_state, actions)
                # print(pred_rewards)
                if clip:
                    pred_rewards = tf.clip_by_value(pred_rewards, min_reward, max_reward)

                logits = self.done_net(current_state, actions)
                # print(logits)
                mask_dist = tfp.distributions.Bernoulli(logits=logits)
                masks *= tf.cast(mask_dist.sample(), tf.float32)
                
                returns += (discount**i) * masks * pred_rewards

                current_state = self.dynamics_net(current_state, actions)
                if clip:
                    current_state = tf.clip_by_value(current_state, min_state, max_state)

                # Collect x, y for plotting
                state_trajectory.append(current_state[0, :2].numpy())
            # print(f"{idx} : {returns}")
            x_coords, y_coords = zip(*state_trajectory)
            plt.plot(x_coords, y_coords, label=f"Sample {idx + 1}", color=colors[idx])

        plt.title("Rollouts for Sampled States")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        # save the image
        plt.savefig(path)

  def estimate_mean_returns(self,
                            initial_states,
                            weights,
                            get_action,
                            discount,
                            min_reward,
                            max_reward,
                            min_state,
                            max_state,
                            clip=True,
                            horizon=1000,
                            settle_timesteps=40):
      """Compute mean and std of returns via rollouts."""
      
      num_trajectories = initial_states.shape[0]
      states = initial_states
      masks = tf.ones((num_trajectories,), dtype=tf.float32)
      
      # List to store rewards for each time step for each trajectory
      all_rewards = []

      for i in range(horizon):
          actions = get_action(states)

          pred_rewards = self.rewards_net(states, actions)
          if clip:
              pred_rewards = tf.clip_by_value(pred_rewards, min_reward, max_reward)
          
          logits = self.done_net(states, actions)
          mask_dist = tfp.distributions.Bernoulli(logits=logits)
          masks *= tf.cast(mask_dist.sample(), tf.float32)

          # Store the rewards for this time step
          if i >= settle_timesteps:
              all_rewards.append(masks * pred_rewards)

          states = self.dynamics_net(states, actions)
          if clip:
              states = tf.clip_by_value(states, min_state, max_state)
      
      # Sum up rewards for each trajectory across time steps
      returns_per_trajectory = tf.reduce_sum(tf.stack(all_rewards, axis=0), axis=0)
      # print(returns_per_trajectory.shape)
      # Calculate the weighted mean and standard deviation of returns
      weighted_returns = weights * returns_per_trajectory
      mean_rewards = tf.reduce_sum(weighted_returns) / tf.reduce_sum(weights)
      std_deviation = tf.math.reduce_std(returns_per_trajectory)
      
      return mean_rewards, std_deviation

  def estimate_returns(self,
                       initial_states,
                       weights,
                       get_action,
                       discount,
                       min_reward,
                       max_reward,
                       min_state,
                       max_state,
                       clip=True,
                       horizon=1000):
    """Compute returns via rollouts.

    Args:
      initial_states: A batch of initial states.
      weights: Weights.
      get_action: An policy.
      discount: MDP discount.
      min_reward: Min reward in the dataset.
      max_reward: Max reward in the dataset.
      min_state: Min state in the dataset.
      max_state: Max state in the dataset.
      clip: Whether to clip obs and rewards.
      horizon: Rollout horizon.

    Returns:
      Estimated returns via rollouts.
    """
    returns = 0
    states = initial_states

    masks = tf.ones((initial_states.shape[0],), dtype=tf.float32)

    for i in range(horizon):
      # print("i", i)
      actions = get_action(states)

      pred_rewards = self.rewards_net(states, actions)
      if clip:
        pred_rewards = tf.clip_by_value(pred_rewards, min_reward,
                                        max_reward)
      logits = self.done_net(states, actions)
      mask_dist = tfp.distributions.Bernoulli(logits=logits)
      masks *= tf.cast(mask_dist.sample(), tf.float32)

      returns += (discount**i) * masks * pred_rewards

      states = self.dynamics_net(states, actions)
      if clip:
        states = tf.clip_by_value(states, min_state, max_state)
    print("pred returns, raw", tf.reduce_sum(
        weights * returns) / tf.reduce_sum(weights))
    return tf.reduce_sum(
        weights * returns) / tf.reduce_sum(weights) * (1 - discount)

