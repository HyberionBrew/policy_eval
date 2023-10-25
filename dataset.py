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

"""Utilities for loading data."""
import pickle
import typing

import numpy as np
import tensorflow as tf


def augment_data(dataset,
                 noise_scale):
  """Augments the data.

  Args:
    dataset: Dictionary with data.
    noise_scale: Scale of noise to apply.

  Returns:
    Augmented data.
  """
  noise_std = np.std(np.concatenate(dataset['rewards'], 0))
  for k, v in dataset.items():
    dataset[k] = np.repeat(v, 3, 0)

  dataset['rewards'][1::3] += noise_std * noise_scale
  dataset['rewards'][2::3] -= noise_std * noise_scale

  return dataset


def weighted_moments(x, weights):
  mean = np.sum(x * weights, 0) / np.sum(weights)
  sqr_diff = np.sum((x - mean)**2 * weights, 0)
  std = np.sqrt(sqr_diff / (weights.sum() - 1))
  return mean, std


class Dataset(object):
  """Dataset class for policy evaluation."""

  def __init__(self,
               data_file_name,
               num_trajectories,
               normalize_states = False,
               normalize_rewards = False,
               eps = 1e-5,
               noise_scale = 0.0,
               bootstrap = True,
               debug=False):
    """Loads data from a file.

    Args:
      data_file_name: filename with data.
      num_trajectories: number of trajectories to select from the dataset.
      normalize_states: whether to normalize the states.
      normalize_rewards: whether to normalize the rewards.
      eps: Epsilon used for normalization.
      noise_scale: Data augmentation noise scale.
      bootstrap: Whether to generated bootstrapped weights.
    """
    with tf.io.gfile.GFile(data_file_name, 'rb') as f:
      dataset = pickle.load(f)

    for k, v in dataset['trajectories'].items():
      dataset['trajectories'][k] = v[:num_trajectories]

    if noise_scale > 0.0:
      dataset['trajectories'] = augment_data(dataset['trajectories'],
                                             noise_scale)

    dataset['trajectories']['steps'] = [
        np.arange(len(state_trajectory))
        for state_trajectory in dataset['trajectories']['states']
    ]

    dataset['initial_states'] = np.stack([
        state_trajectory[0]
        for state_trajectory in dataset['trajectories']['states']
    ])

    num_trajectories = len(dataset['trajectories']['states'])
    if bootstrap:
      dataset['initial_weights'] = np.random.multinomial(
          num_trajectories, [1.0 / num_trajectories] * num_trajectories,
          1).astype(np.float32)[0]
    else:
      dataset['initial_weights'] = np.ones(num_trajectories, dtype=np.float32)

    dataset['trajectories']['weights'] = []
    for i in range(len(dataset['trajectories']['masks'])):
      dataset['trajectories']['weights'].append(
          np.ones_like(dataset['trajectories']['masks'][i]) *
          dataset['initial_weights'][i])

    dataset['initial_weights'] = tf.convert_to_tensor(
        dataset['initial_weights'])
    dataset['initial_states'] = tf.convert_to_tensor(dataset['initial_states'])
    for k, v in dataset['trajectories'].items():
      if 'initial' not in k:
        dataset[k] = tf.convert_to_tensor(
            np.concatenate(dataset['trajectories'][k], axis=0))

    self.states = dataset['states']
    print(self.states[:2])
    self.actions = dataset['actions']
    self.next_states = dataset['next_states']
    self.masks = dataset['masks']
    self.weights = dataset['weights']
    self.rewards = dataset['rewards']
    self.steps = dataset['steps']

    self.initial_states = dataset['initial_states']
    # self.initial_scans = dataset['initial_scans']
    self.initial_weights = dataset['initial_weights']

    self.eps = eps
    self.model_filename = dataset['model_filename']

    if normalize_states:
      self.state_mean = tf.reduce_mean(self.states, 0)
      self.state_std = tf.math.reduce_std(self.states, 0)

      self.initial_states = self.normalize_states(self.initial_states)
      self.states = self.normalize_states(self.states)
      self.next_states = self.normalize_states(self.next_states)
    else:
      self.state_mean = 0.0
      self.state_std = 1.0

    if normalize_rewards:
      self.reward_mean = tf.reduce_mean(self.rewards)
      if tf.reduce_min(self.masks) == 0.0:
        self.reward_mean = tf.zeros_like(self.reward_mean)
      self.reward_std = tf.math.reduce_std(self.rewards)

      self.rewards = self.normalize_rewards(self.rewards)
    else:
      self.reward_mean = 0.0
      self.reward_std = 1.0

  def normalize_states(self, states):
    dtype = tf.convert_to_tensor(states).dtype
    return ((states - self.state_mean) /
            tf.maximum(tf.cast(self.eps, dtype), self.state_std))

  def unnormalize_states(self, states):
    dtype = tf.convert_to_tensor(states).dtype
    return (states * tf.maximum(tf.cast(self.eps, dtype), self.state_std)
            + self.state_mean)

  def normalize_rewards(self, rewards):
    return (rewards - self.reward_mean) / tf.maximum(self.reward_std, self.eps)

  def unnormalize_rewards(self, rewards):
    return rewards * tf.maximum(self.reward_std, self.eps) + self.reward_mean

  def with_uniform_sampling(self, sample_batch_size):
    return tf.data.Dataset.from_tensor_slices(
        (self.states, self.scans, self.actions, self.next_states,self.next_scans, self.rewards, self.masks,
        self.weights, self.log_probs)).repeat().shuffle(
            self.states.shape[0], reshuffle_each_iteration=True).batch(
                sample_batch_size, drop_remainder=True).apply(
      tf.data.experimental.copy_to_device("/gpu:0")).prefetch(tf.data.AUTOTUNE)

  def with_geometric_sampling(self, sample_batch_size,
                              discount):
    """Creates tf dataset with geometric sampling.

    Args:
      sample_batch_size: Batch size for sampling.
      discount: MDP discount.

    Returns:
      TensorFlow dataset.
    """

    sample_weights = discount**tf.cast(self.steps, tf.float32)
    weight_sum = tf.math.cumsum(sample_weights)

    def sample_batch(_):
      values = tf.random.uniform((sample_batch_size,), 0.0,
                                 weight_sum[-1])
      ind = tf.searchsorted(weight_sum, values)
      return (tf.gather(self.states, ind,
                        0), tf.gather(self.actions, ind, 0),
              tf.gather(self.next_states, ind,
                        0), tf.gather(self.rewards, ind, 0),
              tf.gather(self.masks, ind,
                        0), tf.gather(self.weights, ind, 0),
              tf.gather(self.steps, ind, 0))

    return tf.data.experimental.Counter().map(sample_batch).prefetch(tf.data.AUTOTUNE)


class D4rlDataset(Dataset):
  """Dataset class for policy evaluation."""

  # pylint: disable=super-init-not-called
  def __init__(self,
               d4rl_env,
               normalize_states = False,
               normalize_rewards = False,
               eps = 1e-5,
               noise_scale = 0.0,
               bootstrap = True,
               debug=False, 
               path=None):
    """Processes data from D4RL environment.

    Args:
      d4rl_env: gym.Env corresponding to D4RL environment.
      normalize_states: whether to normalize the states.
      normalize_rewards: whether to normalize the rewards.
      eps: Epsilon used for normalization.
      noise_scale: Data augmentation noise scale.
      bootstrap: Whether to generated bootstrapped weights.
    """
    # running on Cpu is necessary since the datasets are to large and I wanted to change as little as possible
    with tf.device('cpu:0'): 
      dataset = dict(
          trajectories=dict(
              states=[],
              actions=[],
              next_states=[],
              rewards=[],
              masks=[]))
      if path is not None:
        d4rl_dataset = d4rl_env.get_dataset(zarr_path=path)
      else:
        d4rl_dataset = d4rl_env.get_dataset()
      dataset_length = len(d4rl_dataset['actions'])
      print(d4rl_dataset['terminals'].shape)
      # print number of 1 terminals
      print(np.sum(d4rl_dataset['terminals']))
      # print amount of timeouts
      print(np.sum(d4rl_dataset['timeouts']))
      print("#########")
      new_trajectory = True
      for idx in range(dataset_length):
        if new_trajectory:
          trajectory = dict(
              states=[], actions=[], next_states=[], rewards=[], masks=[])

        trajectory['states'].append(d4rl_dataset['observations'][idx])
        trajectory['actions'].append(d4rl_dataset['actions'][idx])
        trajectory['rewards'].append(d4rl_dataset['rewards'][idx])
        trajectory['masks'].append(1.0 - d4rl_dataset['terminals'][idx])
        if not new_trajectory:
          trajectory['next_states'].append(d4rl_dataset['observations'][idx])

        end_trajectory = (d4rl_dataset['terminals'][idx] or
                          d4rl_dataset['timeouts'][idx])
        if end_trajectory:
          trajectory['next_states'].append(d4rl_dataset['observations'][idx])
          if d4rl_dataset['timeouts'][idx] and not d4rl_dataset['terminals'][idx]:
            for key in trajectory:
              del trajectory[key][-1]
          if trajectory['actions']:
            for k, v in trajectory.items():
              assert len(v) == len(trajectory['actions'])
              dataset['trajectories'][k].append(np.array(v, dtype=np.float32))
            # print every 200 trajectories
            if len(dataset['trajectories']['actions']) % 600 == 0:
              print('Added trajectory %d with length %d.' % (
                  len(dataset['trajectories']['actions']),
                  len(trajectory['actions'])))
            if debug:
              print('Added trajectory %d with length %d.' % (
                  len(dataset['trajectories']['actions']),
                  len(trajectory['actions'])))
              # break
        new_trajectory = end_trajectory

      if noise_scale > 0.0:
        dataset['trajectories'] = augment_data(dataset['trajectories'],  # pytype: disable=wrong-arg-types  # dict-kwargs
                                              noise_scale)
      #print(dataset['trajectories']['states'].shape)
      #print("üüüüü")
      dataset['trajectories']['steps'] = [
          np.arange(len(state_trajectory))
          for state_trajectory in dataset['trajectories']['states']
      ]

      dataset['initial_states'] = np.stack([
          state_trajectory[0]
          for state_trajectory in dataset['trajectories']['states']
      ])

      num_trajectories = len(dataset['trajectories']['states'])
      if bootstrap:
        dataset['initial_weights'] = np.random.multinomial(
            num_trajectories, [1.0 / num_trajectories] * num_trajectories,
            1).astype(np.float32)[0]
      else:
        dataset['initial_weights'] = np.ones(num_trajectories, dtype=np.float32)

      dataset['trajectories']['weights'] = []
      for i in range(len(dataset['trajectories']['masks'])):
        dataset['trajectories']['weights'].append(
            np.ones_like(dataset['trajectories']['masks'][i]) *
            dataset['initial_weights'][i])

      dataset['initial_weights'] = tf.convert_to_tensor(
          dataset['initial_weights'])
      dataset['initial_states'] = tf.convert_to_tensor(dataset['initial_states'])
      for k, v in dataset['trajectories'].items():
        if 'initial' not in k:
          dataset[k] = tf.convert_to_tensor(
              np.concatenate(dataset['trajectories'][k], axis=0))

      self.states = dataset['states']
      print(self.states[:2])
      #print(self.states.device)
      self.actions = dataset['actions']
      #print(self.actions.device)
      self.next_states = dataset['next_states']
      self.masks = dataset['masks']
      self.weights = dataset['weights']
      self.rewards = dataset['rewards']
      self.steps = dataset['steps']

      self.initial_states = dataset['initial_states']
      self.initial_weights = dataset['initial_weights']

      self.eps = eps
      self.model_filename = None

      if normalize_states:
        # do this on cpu, since it crashes otherwise
        #with tf.device('/CPU:0'): # this needs to be done on CPU since not enough memory
        #print("calculating state mean")
        self.state_mean = tf.reduce_mean(self.states, 0)
        #print("calculating state std")
        # print device of state_mean
        #print(self.state_mean.device)
        self.state_std = tf.math.reduce_std(self.states, 0)

        self.initial_states = self.normalize_states(self.initial_states)
        self.states = self.normalize_states(self.states)
        self.next_states = self.normalize_states(self.next_states)
        # Move the tensors back to GPU
        #with tf.device('/GPU:0'):
        #    self.state_mean = tf.identity(self.state_mean)
        #    self.state_std = tf.identity(self.state_std)
        #    self.initial_states = tf.identity(self.initial_states)
        #    self.states = tf.identity(self.states)
        #    self.next_states = tf.identity(self.next_states)
      else:
        self.state_mean = 0.0
        self.state_std = 1.0

      if normalize_rewards:
        self.reward_mean = tf.reduce_mean(self.rewards)
        if tf.reduce_min(self.masks) == 0.0:
          self.reward_mean = tf.zeros_like(self.reward_mean)
        self.reward_std = tf.math.reduce_std(self.rewards)

        self.rewards = self.normalize_rewards(self.rewards)
      else:
        self.reward_mean = 0.0
        self.reward_std = 1.0
  # pylint: enable=super-init-not-called

class F110Dataset(Dataset):
  """Dataset class for policy evaluation."""

  # pylint: disable=super-init-not-called
  def __init__(self,
               d4rl_env,
               normalize_states = False,
               normalize_rewards = False,
               eps = 1e-5,
               noise_scale = 0.0,
               bootstrap = True,
               debug=False, 
               path=None,
               exclude_agents = [],
               only_agents = [],
               alternate_reward = False,
               scans_as_states=False,
               state_mean=None,
               state_std = None,
               reward_mean = None,
               reward_std = None):
    """Processes data from F110 environment.

    Args:
      d4rl_env: gym.Env corresponding to F110 environment.
      normalize_states: whether to normalize the states.
      normalize_rewards: whether to normalize the rewards.
      eps: Epsilon used for normalization.
      noise_scale: Data augmentation noise scale.
      bootstrap: Whether to generated bootstrapped weights.
    """
    # running on Cpu is necessary since the datasets are to large and I wanted to change as little as possible
    with tf.device('cpu:0'): 
      dataset = dict(
          trajectories=dict(
              states=[],
              scans = [],
              actions=[],
              raw_actions=[],
              next_scans=[],
              next_states=[],
              rewards=[],
              index=[],
              masks=[],
              log_probs = []))
      #print(path)
      
      if path is not None:
        
        d4rl_dataset = d4rl_env.get_dataset(zarr_path=path, 
                                            without_agents=exclude_agents, 
                                            only_agents = only_agents,
                                            alternate_reward = alternate_reward,
                                            #remove_short_trajectories=True,
                                            #split_trajectories=50,
                                            #skip_inital=50,
                                            # min_trajectory_length=600,
                                            )
        print(d4rl_dataset.keys())
      else:
        d4rl_dataset = d4rl_env.get_dataset()
      dataset_length = len(d4rl_dataset['actions'])
      new_trajectory = True
      for idx in range(dataset_length):
        if new_trajectory:
          trajectory = dict(
              states=[], scans=[], actions=[], raw_actions=[], next_states=[], next_scans=[], rewards=[], masks=[], index=[], log_probs=[])
        # print keys of d4rl_dataset
        #print(d4rl_dataset.keys())
        trajectory['states'].append(d4rl_dataset['observations'][idx])
        trajectory['scans'].append(d4rl_dataset['scans'][idx])
        trajectory['index'].append(d4rl_dataset['index'][idx])
        trajectory['actions'].append(d4rl_dataset['actions'][idx])
        trajectory['raw_actions'].append(d4rl_dataset['raw_actions'][idx])
        trajectory['rewards'].append(d4rl_dataset['rewards'][idx])
        trajectory['log_probs'].append(d4rl_dataset['log_probs'][idx])
        trajectory['masks'].append(1.0 - d4rl_dataset['terminals'][idx])
        if not new_trajectory:
          trajectory['next_states'].append(d4rl_dataset['observations'][idx])
          trajectory['next_scans'].append(d4rl_dataset['scans'][idx])

        end_trajectory = (d4rl_dataset['terminals'][idx] or
                          d4rl_dataset['timeouts'][idx])
        if end_trajectory:
          trajectory['next_states'].append(d4rl_dataset['observations'][idx])
          trajectory['next_scans'].append(d4rl_dataset['scans'][idx])

          if d4rl_dataset['timeouts'][idx] and not d4rl_dataset['terminals'][idx]:
            for key in trajectory:
              del trajectory[key][-1]
          if trajectory['actions']:
            #print(trajectory['actions'])
            #print("2-----")
            #print(trajectory.items())
            for k, v in trajectory.items():
              #print(len(v))
              #print(len(trajectory['actions']))
              assert len(v) == len(trajectory['actions'])
              dataset['trajectories'][k].append(np.array(v, dtype=np.float32))
            # print every 200 trajectories
            if len(dataset['trajectories']['actions']) % 600 == 0:
              print('Added trajectory %d with length %d.' % (
                  len(dataset['trajectories']['actions']),
                  len(trajectory['actions'])))
            if debug:
              print('Added trajectory %d with length %d.' % (
                  len(dataset['trajectories']['actions']),
                  len(trajectory['actions'])))
              break
        new_trajectory = end_trajectory

      if noise_scale > 0.0:
        dataset['trajectories'] = augment_data(dataset['trajectories'],  # pytype: disable=wrong-arg-types  # dict-kwargs
                                              noise_scale)
      #print(dataset['trajectories']['states'].shape)
      #print("üüüüü")
      dataset['trajectories']['steps'] = [
          np.arange(len(state_trajectory))
          for state_trajectory in dataset['trajectories']['states']
      ]

      dataset['initial_states'] = np.stack([
          state_trajectory[0]
          for state_trajectory in dataset['trajectories']['states']
      ])
      dataset['initial_scans'] = np.stack([
          state_trajectory[0]
          for state_trajectory in dataset['trajectories']['scans']
      ])

      num_trajectories = len(dataset['trajectories']['states'])
      if bootstrap:
        dataset['initial_weights'] = np.random.multinomial(
            num_trajectories, [1.0 / num_trajectories] * num_trajectories,
            1).astype(np.float32)[0]
      else:
        dataset['initial_weights'] = np.ones(num_trajectories, dtype=np.float32)

      dataset['trajectories']['weights'] = []
      for i in range(len(dataset['trajectories']['masks'])):
        dataset['trajectories']['weights'].append(
            np.ones_like(dataset['trajectories']['masks'][i]) *
            dataset['initial_weights'][i])

      dataset['initial_weights'] = tf.convert_to_tensor(
          dataset['initial_weights'])
      dataset['initial_states'] = tf.convert_to_tensor(dataset['initial_states'])
      dataset['initial_scans'] = tf.convert_to_tensor(dataset['initial_scans'])
      for k, v in dataset['trajectories'].items():
        if 'initial' not in k:
          dataset[k] = tf.convert_to_tensor(
              np.concatenate(dataset['trajectories'][k], axis=0))
      binary_masks = [np.concatenate(([1], np.zeros(len(state_trajectory)-1))) for state_trajectory in dataset['trajectories']['states']]
      self.mask_inital = np.concatenate(binary_masks)
      self.states = dataset['states']
      print(self.states[:2])
      self.index = dataset['index']
      self.scans = dataset['scans']
      self.next_scans = dataset['next_scans']
      # print(self.states.device)
      self.actions = dataset['actions']
      self.raw_actions = dataset['raw_actions']
      #print(self.actions.device)
      # self.initial_scans = dataset['initial_scans']
      self.next_states = dataset['next_states']
      self.masks = dataset['masks']
      self.weights = dataset['weights']
      self.rewards = dataset['rewards']
      self.log_probs = dataset['log_probs']
      self.steps = dataset['steps']

      self.initial_states = dataset['initial_states']
      self.initial_weights = dataset['initial_weights']

      if scans_as_states:
        # raise not impleemnted
        raise NotImplementedError
        #self.states = self.scans
        #self.next_states = self.next_scans
        #self.initial_states = self.initial_scans

      self.eps = eps
      self.model_filename = None

      if normalize_states:
        # do this on cpu, since it crashes otherwise
        #with tf.device('/CPU:0'): # this needs to be done on CPU since not enough memory
        #print("calculating state mean")
        if state_mean is None:
          self.state_mean = tf.reduce_mean(self.states, 0)
          #print("calculating state std")
          # print device of state_mean
          #print(self.state_mean.device)
          self.state_std = tf.math.reduce_std(self.states, 0)
        else:
          self.state_mean = state_mean
          self.state_std = state_std

        self.initial_states = self.normalize_states(self.initial_states)
        self.states = self.normalize_states(self.states)
        self.next_states = self.normalize_states(self.next_states)
        

        # Move the tensors back to GPU
        #with tf.device('/GPU:0'):
        #    self.state_mean = tf.identity(self.state_mean)
        #    self.state_std = tf.identity(self.state_std)
        #    self.initial_states = tf.identity(self.initial_states)
        #    self.states = tf.identity(self.states)
        #    self.next_states = tf.identity(self.next_states)
      else:
        self.state_mean = 0.0
        self.state_std = 1.0

      if normalize_rewards:
        if reward_mean is None:
          self.reward_mean = tf.reduce_mean(self.rewards)
          if tf.reduce_min(self.masks) == 0.0:
            self.reward_mean = tf.zeros_like(self.reward_mean)
          self.reward_std = tf.math.reduce_std(self.rewards)
        else:
          self.reward_mean = reward_mean
          self.reward_std = reward_std

        self.rewards = self.normalize_rewards(self.rewards)
      else:
        self.reward_mean = 0.0
        self.reward_std = 1.0

  #def with_uniform_sampling(self, sample_batch_size):
  #  return tf.data.Dataset.from_tensor_slices(
  #        (self.states, self.scans, self.actions, self.next_states,self.next_scans, self.rewards, self.masks,
  #        self.weights, self.steps)).repeat().shuffle(
  #            self.states.shape[0], reshuffle_each_iteration=True).batch(
  #                sample_batch_size, drop_remainder=True).apply(
  #      tf.data.experimental.copy_to_device("/gpu:0")).prefetch(tf.data.AUTOTUNE)
