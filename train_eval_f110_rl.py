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

r"""Run off-policy evaluation training loop."""

from absl import app
from absl import flags
from absl import logging
from stable_baselines3 import PPO # this needs to be imported before tensorflow
import tensorflow as tf
from utils import plot_reward_heatmap
gpu_memory = 24576 # GPU memory available on the machine
# 45% of the memory, this way we can launch a second process on the same GPU
allowed_gpu_memory = gpu_memory * 0.125
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
      tf.config.experimental.set_virtual_device_configuration(
          gpus[0],
          [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=allowed_gpu_memory)])
  except RuntimeError as e:
      print(e)
print("set gpu memory limit to", allowed_gpu_memory)
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
"""
import gc
import json
import os
import pickle

print	("hi")
import sys
# sys.path.append("/mnt/hdd2/fabian/demo/ws_ope/policy_eval")


# import d4rl  # pylint: disable=unused-import
import f110_gym
import f110_orl_dataset
import gymnasium as gym

print	("hi")
from gymnasium.wrappers import time_limit
import numpy as np

# from tf_agents.environments import gym_wrapper
# from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
import tqdm

from policy_eval import utils_f110 as utils
from policy_eval.actor import Actor
from policy_eval.behavior_cloning import BehaviorCloning
from policy_eval.dataset import F110Dataset
from policy_eval.dataset import Dataset
from policy_eval.dual_dice import DualDICE
from policy_eval.model_based import ModelBased
from policy_eval.q_fitter import QFitter
from model_based_2 import ModelBased2
from ftg_agents.agents import *
from tensorboardX import SummaryWriter

from f110_orl_dataset.normalize_dataset import Normalize
from f110_orl_dataset.dataset_agents import F110Actor,F110Stupid

import os
import sys
import torch


EPS = np.finfo(np.float32).eps
FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'f110_gym', 'Name of the environment.')
flags.DEFINE_bool('f110', True, 'Whether to use Trifinger envs and datasets.')
flags.DEFINE_string('target_policy', 'velocity', 'Name of target agent')
flags.DEFINE_float('speed', 1.0, 'Mean speed of the car, for the agent') 
#flags.DEFINE_string('d4rl_policy_filename', None,
#                    'Path to saved pickle of D4RL policy.')
#flags.DEFINE_string('trifinger_policy_class', "trifinger_rl_example.example.TorchPushPolicy",
#                    'Policy class name for Trifinger.')
flags.DEFINE_bool('load_mb_model', False, 'Whether to load a model-based model.')
flags.DEFINE_integer('seed', 1, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 3e-5, 'Critic learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('behavior_policy_std', None,
                   'Noise scale of behavior policy.')
flags.DEFINE_float('target_policy_std', 0.0, 'Noise scale of target policy.')
flags.DEFINE_bool('target_policy_noisy', False, 'inject noise into the actions of the target policy')
# flags.DEFINE_integer('num_trajectories', 1000, 'Number of trajectories.') # this is not actually used
flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')

flags.DEFINE_integer('num_updates', 1_000_000, 'Number of updates.')
flags.DEFINE_integer('eval_interval', 10_000, 'Logging interval.')
flags.DEFINE_integer('log_interval', 10_000, 'Logging interval.')
flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
flags.DEFINE_float('tau', 0.005,
                   'Soft update coefficient for the target network.')
flags.DEFINE_string('save_dir', '/app/ws/logdir',
                    'Directory to save results to.')
flags.DEFINE_string(
    'data_dir',
    '/tmp/policy_eval/trajectory_datasets/',
    'Directory with data for evaluation.')
flags.DEFINE_boolean('normalize_states', True, 'Whether to normalize states.')
flags.DEFINE_boolean('normalize_rewards', True, 'Whether to normalize rewards.')
flags.DEFINE_boolean('bootstrap', True,
                     'Whether to generated bootstrap weights.')
flags.DEFINE_enum('algo', 'fqe', ['fqe', 'dual_dice', 'mb', 'iw', 'dr'],
                  'Algorithm for policy evaluation.')
flags.DEFINE_float('noise_scale', 0.0, 'Noise scaling for data augmentation.') # 0.25
flags.DEFINE_string('model_path', None, 'Path to saved model.')
flags.DEFINE_bool('no_behavior_cloning', False, 'Whether to use behavior cloning')
flags.DEFINE_bool('alternate_reward', False, 'Whether to use alternate reward')
flags.DEFINE_string('path', "trajectories.zarr", "The reward dataset to use")
flags.DEFINE_integer('clip_trajectory_max', 0, 'Max trajectory length')


def make_hparam_string(json_parameters=None, **hparam_str_dict):
  if json_parameters:
    for key, value in json.loads(json_parameters).items():
      if key not in hparam_str_dict:
        hparam_str_dict[key] = value
  return ','.join([
      '%s=%s' % (k, str(hparam_str_dict[k]))
      for k in sorted(hparam_str_dict.keys())
  ])


def main(_):
  
  np.random.seed(FLAGS.seed)
  # assert not FLAGS.d4rl and FlAGS.trifinger

  """
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Restrict TensorFlow to only allocate 20% of the memory on the first GPU
      tf.config.experimental.set_virtual_device_configuration(
          gpus[0],
          [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(allowed_gpu_memory))])
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Virtual devices must be set before GPUs have been initialized
      print(e, " GPUs must be set at program startup")
  
  """
  # set the correct save experiment directory
  experiment_directory = "1411"
  save_directory = os.path.join(FLAGS.save_dir, experiment_directory)
  if not os.path.exists(save_directory):
    os.makedirs(save_directory)
  # now the algo directory
  save_directory = os.path.join(save_directory, f"{FLAGS.algo}")
  if not os.path.exists(save_directory):
    os.makedirs(save_directory)
  # path
  save_directory = os.path.join(save_directory, f"{FLAGS.path}")
  if not os.path.exists(save_directory):
    os.makedirs(save_directory)

  #now the max_timesteps directory
  save_directory = os.path.join(save_directory, f"{FLAGS.clip_trajectory_max}")
  if not os.path.exists(save_directory):
    os.makedirs(save_directory)
  # now the target policy directory
  save_directory = os.path.join(save_directory, f"{FLAGS.target_policy}")
  if not os.path.exists(save_directory):
    os.makedirs(save_directory)
  
  # create the directory if it does not exist

  tf.random.set_seed(FLAGS.seed)
  # import f110
  

  # get current system time
  import datetime
  now = datetime.datetime.now()
  time = now.strftime("%Y-%m-%d-%H-%M-%S")

  hparam_str = make_hparam_string(
      seed=FLAGS.seed, env_name=FLAGS.env_name, algo=FLAGS.algo,
      target_policy=FLAGS.target_policy, 
      std=FLAGS.target_policy_std, time=time, target_policy_noisy=FLAGS.target_policy_noisy, noise_scale=FLAGS.noise_scale)
  
  
  summary_writer = tf.summary.create_file_writer(
      os.path.join(save_directory, hparam_str))
  summary_writer.set_as_default()

  if FLAGS.f110:
    subsample_laser = 20
    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of right now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1,
            params=dict(vmin=0.5, vmax=2.0)),
              render_mode="human")
    )
    env = F110Env
    # print available classes of env
    print(env.observation_space)
    
    print("-------")
    clip_trajectory_length = None
    if FLAGS.clip_trajectory_max > 0:
      clip_trajectory_length = (0,FLAGS.clip_trajectory_max)
    behavior_dataset = F110Dataset(
        env,
        normalize_states=FLAGS.normalize_states,
        normalize_rewards=FLAGS.normalize_rewards,
        noise_scale=FLAGS.noise_scale,
        bootstrap=FLAGS.bootstrap,
        debug=False,
        path = f"/app/ws/f1tenth_orl_dataset/data/{FLAGS.path}", #trajectories.zarr",
        exclude_agents = ['progress_weight', 'raceline_delta_weight', 'min_action_weight'],#['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
        scans_as_states=False,
        alternate_reward=FLAGS.alternate_reward,
        include_timesteps_in_obs = True,
        only_terminals=True,
        clip_trajectory_length= clip_trajectory_length,
        )
    
    """
    eval1_dataset = F110Dataset(
        env,
        normalize_states=FLAGS.normalize_states,
        normalize_rewards=FLAGS.normalize_rewards,
        noise_scale=FLAGS.noise_scale,
        bootstrap=FLAGS.bootstrap,
        debug=False,
        path = f"/app/ws/f1tenth_orl_dataset/data/{FLAGS.path}", #trajectories.zarr",
        exclude_agents = ,#['det'] + [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
        scans_as_states=False,
        alternate_reward=FLAGS.alternate_reward,)
    
    
    eval2_dataset = F110Dataset(
      env,
      normalize_states=FLAGS.normalize_states,
      normalize_rewards=FLAGS.normalize_rewards,
      noise_scale=FLAGS.noise_scale,
      bootstrap=FLAGS.bootstrap,
      debug=False,
      path = f"/app/ws/f1tenth_orl_dataset/data/{FLAGS.path}", #trajectories.zarr",
      only_agents= [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
      scans_as_states=False,
      alternate_reward=FLAGS.alternate_reward,)
    """
    # sample random 1500 states from both eval1 and eval2
    # eval1_dataset_states = eval1_dataset.states
    #print("eval1_dataset_states", eval1_dataset_states.shape)
    #eval2_dataset_states = eval2_dataset.states
    #print("eval2_dataset_states", eval2_dataset_states.shape)
    #sampled_eval1_states = tf.random.shuffle(eval1_dataset_states)[:1500]
    #sampled_eval2_states = tf.random.shuffle(eval2_dataset_states)[:1500]
    #print("sampled_eval1_states", sampled_eval1_states.shape)
    #print("sampled_eval1_states", sampled_eval1_states[:2])
    print("Finished loading F110 Dataset")
    print(behavior_dataset.initial_states.shape)
  else:
    # Throw unsupported error
    # Trifinger and other envs cannot be run in the same script since there are some incompatibilities
    raise NotImplementedError
  tf_dataset = behavior_dataset.with_uniform_sampling(FLAGS.sample_batch_size)
  tf_dataset_iter = iter(tf_dataset)

  if FLAGS.f110:
    # model = PPO.load(FLAGS.model_path)
    # print("yo")
    actor = F110Actor(FLAGS.target_policy, deterministic=False) #F110Stupid()
    model_input_normalizer = Normalize()
    # print("no")
    # actor = model
  else:
    # Throw unsupported error
    raise NotImplementedError

  if 'fqe' in FLAGS.algo or 'dr' in FLAGS.algo:
    print("SPEC", behavior_dataset.states.shape[1])
    print(env.action_spec().shape)
    model = QFitter(behavior_dataset.states.shape[1],#env.observation_spec().shape[0],
                    env.action_spec().shape[1], FLAGS.lr, FLAGS.weight_decay,
                    FLAGS.tau, 
                    use_time=True, 
                    timestep_constant = behavior_dataset.timestep_constant)

  elif 'mb' in FLAGS.algo:
    #model = ModelBased(behavior_dataset.states.shape[1], #env.observation_spec().shape[0],
    #                   env.action_spec().shape[1], learning_rate=FLAGS.lr,
    #                   weight_decay=FLAGS.weight_decay)
    min_state = tf.reduce_min(behavior_dataset.states, 0)
    max_state = tf.reduce_max(behavior_dataset.states, 0)
    writer = SummaryWriter(log_dir= os.path.join(FLAGS.save_dir, f"mb_latest", "ensemble_"+hparam_str))
    model = ModelBased2(behavior_dataset.states.shape[1],
                      env.action_spec().shape[1], [256,256,256,256], 
                      dt=1/20, 
                      min_state=min_state, 
                      max_state=max_state, logger=writer, 
                      dataset=behavior_dataset,
                      learning_rate=FLAGS.lr,
                      weight_decay=FLAGS.weight_decay,
                      target_reward=FLAGS.path,)
    #if FLAGS.load_mb_model:
    #  model.load("/app/ws/logdir/mb/mb_model_250000")
    if FLAGS.load_mb_model:
      model.load("/app/ws/logdir/mb/mb_model_110000", "new_model")
  
  elif 'dual_dice' in FLAGS.algo:
    model = DualDICE(behavior_dataset.states.shape[1],#env.observation_spec().shape[0],
                     env.action_spec().shape[1], FLAGS.weight_decay)
  if 'iw' in FLAGS.algo or 'dr' in FLAGS.algo:
    if FLAGS.no_behavior_cloning:
      behavior = None
    else:
      action_spec = gym.spaces.Box(low=np.asarray([-1.0,-1.0]),high=np.asarray([1.0,1.0]),shape=(2,))
      behavior = BehaviorCloning(behavior_dataset.states.shape[1],#env.observation_spec().shape[0],
                                action_spec,
                                FLAGS.lr, FLAGS.weight_decay)
    
  #@tf.function
  #@tf.function
  def get_target_actions(states, scans= None, batch_size=5000):
    num_batches = int(np.ceil(len(states) / batch_size))
    actions_list = []
    # batching, s.t. we dont run OOM
    for i in range(num_batches):
      start_idx = i * batch_size
      end_idx = min((i + 1) * batch_size, len(states))
      batch_states = states[start_idx:end_idx]

      # unnormalize from the dope dataset normalization
      batch_states_unnorm = behavior_dataset.unnormalize_states(batch_states) # this needs batches
      batch_states_unnorm = batch_states_unnorm.numpy()

      # get scans
      if scans is not None:
        laser_scan = scans[start_idx:end_idx]
      else:
        laser_scan = F110Env.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
        laser_scan = model_input_normalizer.normalize_laser_scan(laser_scan)

      # back to dict
      model_input_dict = model_input_normalizer.unflatten_batch(batch_states_unnorm)
      # normalize back to model input
      model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
     
      # now also append the laser scan
      model_input_dict['lidar_occupancy'] = laser_scan

      batch_actions = actor(
        model_input_dict,
        std=FLAGS.target_policy_std)[1]
      
      actions_list.append(batch_actions)

    actions = tf.concat(actions_list, axis=0)
    actions = tf.convert_to_tensor(actions)
    return actions

  #@tf.function
  def get_target_logprobs(states, actions, scans=None, batch_size=5000):
    num_batches = int(np.ceil(len(states) / batch_size))
    log_probs_list = []
    # print(num_batches)
    for i in range(num_batches):
      # print(i)
      # Calculate start and end indices for the current batch
      start_idx = i * batch_size
      end_idx = min((i + 1) * batch_size, len(states))
      # Extract the current batch of states
      batch_states = states[start_idx:end_idx]
      batch_states_unnorm = behavior_dataset.unnormalize_states(batch_states)
      
      # Extract the current batch of actions
      batch_actions = actions[start_idx:end_idx]

      # get scans
      if scans is not None:
        laser_scan = scans[start_idx:end_idx]
      else:
        laser_scan = F110Env.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
        laser_scan = model_input_normalizer.normalize_laser_scan(laser_scan)

      # back to dict
      model_input_dict = model_input_normalizer.unflatten_batch(batch_states_unnorm)
      # normalize back to model input
      model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
      # now also append the laser scan
      model_input_dict['lidar_occupancy'] = laser_scan

      # Compute log_probs for the current batch
      batch_log_probs = actor(
          model_input_dict,
          actions=batch_actions,
          std=FLAGS.target_policy_std)[2]
      
      # Sum along the last axis if the rank is greater than 1
      if tf.rank(batch_log_probs) > 1:
          batch_log_probs = tf.reduce_sum(batch_log_probs, -1)
      
      # Collect the batch_log_probs
      log_probs_list.append(batch_log_probs)
    # Concatenate the collected log_probs from all batches
    log_probs = tf.concat(log_probs_list, axis=0)
    
    # Convert to TensorFlow tensor
    log_probs = tf.convert_to_tensor(log_probs)
    #print("log_probs", log_probs.shape)
    return log_probs

  min_reward = tf.reduce_min(behavior_dataset.rewards)
  max_reward = tf.reduce_max(behavior_dataset.rewards)
  min_state = tf.reduce_min(behavior_dataset.states, 0)
  max_state = tf.reduce_max(behavior_dataset.states, 0)

  #@tf.function
  def update_step():
    import time
    # start = time.time()
    # (self.states, self.scans, self.actions, self.next_states,self.next_scans, 
    # self.rewards, self.masks,
    #     self.weights, self.log_prob)
    (states, scans, actions, next_states, next_scans, rewards, masks, weights,
     log_prob, timesteps) = next(tf_dataset_iter)

    # print("max_trajectory_length ", np.max(behavior_dataset.steps) + 1)
    #print("--------")
    #print("scan on record", scans[:2])
    # get_target_actions(sta)
    #print(".........")
    #states = 
    #print(states[:2])
    # print("action dataset:", actions[:2].shape)
    #actions = get_target_actions(states[:2], scans=scans[:2])
    #log_prob = get_target_logprobs(states[:2], actions[:2], scans=scans[:2])
    #print("action infered:", actions)
    #print("log_prob infered:", log_prob)
    #exit()
    # print(rewards.shape)
    # print(rewards[0])
    if 'fqe' in FLAGS.algo or 'dr' in FLAGS.algo or 'hybrid' in FLAGS.algo:
      next_actions = get_target_actions(next_states, scans=next_scans)

      model.update(states, actions, next_states, next_actions, rewards, masks,
                   weights, FLAGS.discount, min_reward, max_reward, timesteps=timesteps)

    elif 'mb' in FLAGS.algo or 'hybrid' in FLAGS.algo:
      if not(FLAGS.load_mb_model):
        model.update(states, actions, next_states, rewards, masks,
                   weights)

    elif 'dual_dice' in FLAGS.algo:
      initial_actions = get_target_actions(behavior_dataset.initial_states)
      next_actions = get_target_actions(next_states)
      model.update(behavior_dataset.initial_states, initial_actions,
                   behavior_dataset.initial_weights, states, actions,
                   next_states, next_actions, masks, weights, FLAGS.discount)

    if 'iw' in FLAGS.algo or 'dr' in FLAGS.algo:
      if FLAGS.no_behavior_cloning:
        pass
      else:
        # print(states.shape)
        # print(actions.shape)
        #actions_behavior = tf.expand_dims(actions, axis=1)
        #print(states.shape)
        #print(actions_behavior.shape)
        behavior.update(states, actions, weights)
    end = time.time()
    # print("Time to update model: ", end-start)
  gc.collect()

  for i in tqdm.tqdm(range(FLAGS.num_updates), desc='Running Training',  mininterval=5.0):
    if not FLAGS.load_mb_model:
      # skip if we load a mb model
      update_step()

    if i % FLAGS.eval_interval == 0:
      if 'fqe' in FLAGS.algo:
        pred_returns,_ = model.estimate_returns(behavior_dataset.initial_states,
                                              behavior_dataset.initial_weights,
                                              get_target_actions)
        print(behavior_dataset.times)
        #pred_returns_fqe_mb = MBFQERollout(model_mb, 
        #                                   model_fqe, 
        #                                   behavior_dataset, 
        #                                   get_target_actions, 
        #                                   discount=FLAGS.discount,
        #                                   horizon = 10,
        #                                   dt = behavior_dataset.timestep_constant)

        if (i % FLAGS.eval_interval*2) == 0:
          model.save(f"/app/ws/logdir/{FLAGS.target_policy}/fqe_model_{i}_{FLAGS.clip_trajectory_max}_{FLAGS.target_policy}")
      elif 'hybrid' in FLAGS.algo:
          """
          forward_inital_states = mb_model.sim_steps(
            behavior_dataset.states,
            get_target_actions,
            FLAGS.discount,
            min_state, max_state, 
            horizon= 50,
          )
          pred_returns,_ = fqe_model.estimate_returns(forward_inital_states,
                                              behavior_dataset.initial_weights,
                                              get_target_actions)
          
          """
      elif 'mb' in FLAGS.algo:
        horizon = FLAGS.clip_trajectory_max      
        
        pred_returns, std = model.estimate_returns(behavior_dataset.initial_states,
                        behavior_dataset.initial_weights,
                        get_target_actions, horizon=horizon,
                        discount=FLAGS.discount,)
        print(pred_returns)
        print(std)
        #exit()
        #pred_returns = behavior_dataset.unnormalize_rewards(pred_returns)
        #std = behavior_dataset.unnormalize_rewards(std)
        tf.summary.scalar('train/pred returns', pred_returns, step=i)
        tf.summary.scalar('train/pred std', std, step=i)
        
        model.plot_rollouts_fast(behavior_dataset,
                                behavior_dataset.unnormalize_rewards,
                      horizon= horizon,
                      num_samples=15,
                      get_target_action=get_target_actions,
                      path = f"logdir/plts/mb/mb_rollouts_{FLAGS.path}_{FLAGS.target_policy}_{FLAGS.discount}_{i}_torch.png")#np.max(behavior_dataset.steps) + 1)


      elif FLAGS.algo in ['dual_dice']:
        pred_returns, pred_ratio = model.estimate_returns(iter(tf_dataset))

        tf.summary.scalar('train/pred ratio', pred_ratio, step=i)
      elif 'iw' in FLAGS.algo or 'dr' in FLAGS.algo:
        discount = FLAGS.discount
        if FLAGS.no_behavior_cloning:
          behavior_log_probs = behavior_dataset.log_probs
        else:
          _, behavior_log_probs = behavior(behavior_dataset.states,
                                          behavior_dataset.actions)
        print("target log probs")
        target_log_probs = get_target_logprobs(behavior_dataset.states,
                                               behavior_dataset.actions,
                                               scans=behavior_dataset.scans)
        
        print("got target_log_probs")
        # print(target_log_probs.shape)
        offset = 0.0
        rewards = behavior_dataset.rewards
        if 'dr' in FLAGS.algo:
          # Doubly-robust is effectively the same as importance-weighting but
          # transforming rewards at (s,a) to r(s,a) + gamma * V^pi(s') -
          # Q^pi(s,a) and adding an offset to each trajectory equal to V^pi(s0).
          print("estimating returns")
          offset, std_deviation = model.estimate_returns(behavior_dataset.initial_states,
                                          behavior_dataset.initial_weights,
                                          get_target_actions)
          all_returns = model.estimate_returns_unweighted(behavior_dataset.states, get_target_actions)
          model.save(f"/app/ws/logdir/{FLAGS.target_policy}/fqe_model_{i}")
          # save image of 
          tf.summary.scalar('train/pred returns (fqe)', behavior_dataset.unnormalize_rewards(offset), step=i)
          tf.summary.scalar('std_deviation returns (fqe)', behavior_dataset.unnormalize_rewards(std_deviation), step=i)
          plot_reward_heatmap(behavior_dataset.states.cpu().numpy(), all_returns.cpu().numpy(), bins=150, name=f"logdir/plts/fqe_{i}_{FLAGS.target_policy}_{FLAGS.discount}")
          print("fqe std_deviation", behavior_dataset.unnormalize_rewards(std_deviation))
          #print("q values")
          q_values = (model(behavior_dataset.states, behavior_dataset.actions, 
                            timesteps=behavior_dataset.timesteps) /
                      (1 - discount))
          #print("got q_vals")
          n_samples = 10
          next_actions = [get_target_actions(behavior_dataset.next_states, scans=behavior_dataset.next_scans)
                          for _ in range(n_samples)]
          #print("next q_vals")
          next_q_values = sum(
              [model(behavior_dataset.next_states, next_action, 
                     timesteps=behavior_dataset.timesteps+behavior_dataset.timestep_constant) / (1 - discount)
               for next_action in next_actions]) / n_samples
          print("got next q_vals")
          rewards = rewards + discount * next_q_values - q_values

        # Now we compute the self-normalized importance weights.
        # Self-normalization happens over trajectories per-step, so we
        # restructure the dataset as [num_trajectories, num_steps].
        num_trajectories = len(behavior_dataset.initial_states)
        max_trajectory_length = np.max(behavior_dataset.steps) + 1
        print("max, len", max_trajectory_length)
        trajectory_weights = behavior_dataset.initial_weights
        trajectory_starts = np.where(np.equal(behavior_dataset.steps, 0))[0]

        batched_rewards = np.zeros([num_trajectories, max_trajectory_length])
        batched_masks = np.zeros([num_trajectories, max_trajectory_length])
        batched_log_probs = np.zeros([num_trajectories, max_trajectory_length])

        for traj_idx, traj_start in enumerate(trajectory_starts):
          traj_end = (trajectory_starts[traj_idx + 1]
                      if traj_idx + 1 < len(trajectory_starts)
                      else len(rewards))
          traj_length = traj_end - traj_start
          batched_rewards[traj_idx, :traj_length] = rewards[traj_start:traj_end]
          batched_masks[traj_idx, :traj_length] = 1.
          batched_log_probs[traj_idx, :traj_length] = (
              -behavior_log_probs[traj_start:traj_end] +
              target_log_probs[traj_start:traj_end])

        batched_weights = (batched_masks *
                           (discount **
                            np.arange(max_trajectory_length))[None, :])

        clipped_log_probs = np.clip(batched_log_probs, -6., 2.)
        cum_log_probs = batched_masks * np.cumsum(clipped_log_probs, axis=1)
        cum_log_probs_offset = np.max(cum_log_probs, axis=0)
        cum_probs = np.exp(cum_log_probs - cum_log_probs_offset[None, :])
        avg_cum_probs = (
            np.sum(cum_probs * trajectory_weights[:, None], axis=0) /
            (1e-10 + np.sum(batched_masks * trajectory_weights[:, None],
                            axis=0)))
        norm_cum_probs = cum_probs / (1e-10 + avg_cum_probs[None, :])

        weighted_rewards = batched_weights * batched_rewards * norm_cum_probs
        trajectory_values = np.sum(weighted_rewards, axis=1)
        avg_trajectory_value = ((1 - discount) *
                                np.sum(trajectory_values * trajectory_weights) /
                                np.sum(trajectory_weights))
        pred_returns = offset + avg_trajectory_value
      if not(FLAGS.algo == 'mb'):
        pred_returns = behavior_dataset.unnormalize_rewards(pred_returns)

        tf.summary.scalar('train/pred returns', pred_returns, step=i)
      logging.info('pred returns=%f', pred_returns)


app.run(main)
if __name__ == '__main__':
  print("Running main")
  app.run(main)
