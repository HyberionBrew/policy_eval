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

"""Policy evaluation with TD learning."""
import typing
import tensorflow as tf
from tensorflow_addons import optimizers as tfa_optimizers
from policy_eval import keras_utils


def soft_update(net,
                target_net,
                tau = 0.005):
  for var, target_var in zip(net.variables, target_net.variables):
    new_value = var * tau + target_var * (1 - tau)
    target_var.assign(new_value)


class CriticNet(tf.keras.Model):
  """A critic network that estimates a dual Q-function."""

  def __init__(self, state_dim, action_dim):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
    """
    super(CriticNet, self).__init__()
    self.critic = keras_utils.create_mlp(state_dim + action_dim, 1, [256, 256])

  def call(self, states, actions):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    x = tf.concat([states, actions], -1)
    return tf.squeeze(self.critic(x), 1)


class QFitter(object):
  """A critic network that estimates a dual Q-function."""

  def __init__(self, state_dim, action_dim, critic_lr,
               weight_decay, tau, use_time=False, timestep_constant = 0.01, log_frequency = 500):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      critic_lr: Critic learning rate.
      weight_decay: Weight decay.
      tau: Soft update discount.
    """

    if use_time:
      state_dim += 1
      self.use_time = True
      self.timestep_constant = timestep_constant
    
    self.critic = CriticNet(state_dim, action_dim)
    self.critic_target = CriticNet(state_dim, action_dim)

    self.tau = tau
    soft_update(self.critic, self.critic_target, tau=1.0)

    self.optimizer = tfa_optimizers.AdamW(learning_rate=critic_lr,
                                          weight_decay=weight_decay)
    self.log_frequency = log_frequency

  def __call__(self, states, actions, batch_size = 1000):
    # batch this call to avoid OOM
    n_data = states.shape[0]
    # List to store results
    results = []

    # Process in batches
    for i in range(0, n_data, batch_size):
      batch_states = states[i: min(i + batch_size, n_data)]
      batch_actions = actions[i: min(i + batch_size,n_data)]

      # Call the critic_target function for the batch
      batch_result = self.critic_target(batch_states, batch_actions)
      # Store the result
      results.append(batch_result)

    # Concatenate all batch results
    final_result = tf.concat(results, axis=0)
    return final_result

  @tf.function
  def update(self, states, actions,
             next_states, next_actions,
             rewards, masks, weights,
             discount, min_reward,
             max_reward, timesteps, timestep_constant):
    """Updates critic parameters.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      next_states: A batch of next states.
      next_actions: A batch of next actions.
      rewards: A batch of rewards.
      masks: A batch of masks indicating the end of the episodes.
      weights: A batch of weights.
      discount: An MDP discount factor.
      min_reward: min reward in the dataset.
      max_reward: max reward in the dataset.

    Returns:
      Critic loss.
    """
    if self.use_time:
      states = tf.concat([states, timesteps], axis = 1)
      next_states = tf.concat([next_states, timesteps + timestep_constant], axis = 1)
    

    next_q = self.critic_target(next_states, next_actions) / (1 - discount)
    target_q = rewards + discount * masks * next_q
    target_q = tf.clip_by_value(target_q, min_reward / (1 - discount),
                                max_reward / (1 - discount))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.trainable_variables)
      q = self.critic(states, actions) / (1 - discount)
      critic_loss = (
          tf.reduce_sum(tf.square(target_q - q) * weights) /
          tf.reduce_sum(weights))

    critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

    self.optimizer.apply_gradients(
        zip(critic_grads, self.critic.trainable_variables))

    soft_update(self.critic, self.critic_target, self.tau)
    if self.optimizer.iterations % self.log_frequency == 0:
      tf.summary.scalar('train/critic loss', critic_loss,
                        step=self.optimizer.iterations)
    return critic_loss

  #@tf.function
  def estimate_returns(
      self, initial_states, initial_weights,
      get_action):
    """Estimate returns with fitted q learning.

    Args:
      initial_states: Initial states.
      initial_weights: Weights for the initial states.
      get_action: Policy function.

    Returns:
      Estimate of returns.
    """
    initial_actions = get_action(initial_states)
    preds = self(initial_states, initial_actions)
    # print(preds)
    weighted_mean = (tf.reduce_sum(preds * initial_weights) /
            tf.reduce_sum(initial_weights))
    weighted_variance = tf.reduce_sum(initial_weights * tf.square(preds - weighted_mean)) / tf.reduce_sum(initial_weights)
    weighted_stddev = tf.sqrt(weighted_variance)
    return weighted_mean, weighted_stddev
  
  def estimate_returns_unweighted(self, initial_states,
      get_action):
      initial_actions = get_action(initial_states)
      preds = self(initial_states, initial_actions)
      return preds
  def save(self, path):
    self.critic.save(path + '/critic')
    self.critic_target.save(path + '/critic_target')
    
  def load(self, path):
    tf.keras.models.load_model(path + '/critic_target')
    tf.keras.models.load_model(path + '/critic')