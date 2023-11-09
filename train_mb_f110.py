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
from model_based_2 import ModelBased2
from utils import plot_reward_heatmap
gpu_memory = 12209 # GPU memory available on the machine
# 45% of the memory, this way we can launch a second process on the same GPU
allowed_gpu_memory = gpu_memory * 0.25
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
from tensorboardX import SummaryWriter
from ftg_agents.agents import *

from f110_orl_dataset.normalize_dataset import Normalize
from f110_orl_dataset.dataset_agents import F110Actor,F110Stupid

import os
import sys
import torch


EPS = np.finfo(np.float32).eps
FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'f110_gym', 'Name of the environment.')
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
flags.DEFINE_float('noise_scale', 0.0, 'Noise scaling for data augmentation.') # 0.25
flags.DEFINE_string('model_path', None, 'Path to saved model.')
flags.DEFINE_bool('no_behavior_cloning', False, 'Whether to use behavior cloning')
flags.DEFINE_bool('alternate_reward', False, 'Whether to use alternate reward')
flags.DEFINE_string('path', "trajectories.zarr", "The reward dataset to use")
flags.DEFINE_bool('use_torch', False, 'Whether to use torch (which is the new model)')
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
  tf.random.set_seed(FLAGS.seed)
  # import f110
  use_torch = FLAGS.use_torch

  # get current system time
  import datetime
  now = datetime.datetime.now()
  time = now.strftime("%Y-%m-%d-%H-%M-%S")

  hparam_str = make_hparam_string(
      seed=FLAGS.seed, env_name=FLAGS.env_name, algo='mb',
      target_policy=FLAGS.target_policy, 
      std=FLAGS.target_policy_std, time=time, target_policy_noisy=FLAGS.target_policy_noisy, noise_scale=FLAGS.noise_scale)

  if use_torch:
    writer = SummaryWriter(log_dir= os.path.join(FLAGS.save_dir, f"f110_rl_{FLAGS.discount}_mb_{FLAGS.path}_0211", "ensemble_"+hparam_str))
  else:
      summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, f"f110_rl_{FLAGS.discount}_mb_{FLAGS.path}_test_0211", "ensemble_"+hparam_str))
      summary_writer.set_as_default()
  subsample_laser = 20
  F110Env = gym.make('f110_with_dataset-v0',
  # only terminals are available as of right now 
      **dict(name='f110_with_dataset-v0',
          config = dict(map="Infsaal", num_agents=1,
          params=dict(vmin=0.5, vmax=2.0)),
            render_mode="human")
  )
  env = F110Env

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
      include_timesteps_in_obs = True,)
  eval_datasets = []
  
  eval_agents = ['progress_weight', 'raceline_delta_weight', 'min_action_weight']
  print("means and stds")
  print(behavior_dataset.reward_mean, behavior_dataset.reward_std,
        behavior_dataset.state_mean,
      behavior_dataset.state_std,)
  if False:
      evaluation_dataset = F110Dataset(
        env,
        normalize_states=FLAGS.normalize_states,
        normalize_rewards=FLAGS.normalize_rewards,
        noise_scale=FLAGS.noise_scale,
        bootstrap=FLAGS.bootstrap,
        debug=False,
        path = f"/app/ws/f1tenth_orl_dataset/data/trajectories_test.zarr", #trajectories.zarr",
         #['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
        scans_as_states=False,
        alternate_reward=FLAGS.alternate_reward,
        include_timesteps_in_obs = True,
        reward_mean = behavior_dataset.reward_mean,
        reward_std = behavior_dataset.reward_std,
        state_mean = behavior_dataset.state_mean,
        state_std = behavior_dataset.state_std,
        )
      eval_datasets.append(evaluation_dataset)
  
  if True:
    for i, agent in enumerate(eval_agents):
      evaluation_dataset = F110Dataset(
        env,
        normalize_states=FLAGS.normalize_states,
        normalize_rewards=FLAGS.normalize_rewards,
        noise_scale=FLAGS.noise_scale,
        bootstrap=FLAGS.bootstrap,
        debug=False,
        path = f"/app/ws/f1tenth_orl_dataset/data/{FLAGS.path}", #trajectories.zarr",
        only_agents = [agent], #['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
        scans_as_states=False,
        alternate_reward=FLAGS.alternate_reward,
        include_timesteps_in_obs = True,
        reward_mean = behavior_dataset.reward_mean,
        reward_std = behavior_dataset.reward_std,
        state_mean = behavior_dataset.state_mean,
        state_std = behavior_dataset.state_std,
        )
      eval_datasets.append(evaluation_dataset)

  print("Finished loading F110 Dataset")
  
  tf_dataset = behavior_dataset.with_uniform_sampling(FLAGS.sample_batch_size)
  tf_dataset_iter = iter(tf_dataset)

  if use_torch:
    min_state = tf.reduce_min(behavior_dataset.states, 0)
    max_state = tf.reduce_max(behavior_dataset.states, 0)
    model = ModelBased2(behavior_dataset.states.shape[1],
                      env.action_spec().shape[1], [256,256,256,256], 
                      dt=1/20, 
                      min_state=min_state, 
                      max_state=max_state, logger=writer, 
                      dataset=behavior_dataset,
                      learning_rate=FLAGS.lr,
                      weight_decay=FLAGS.weight_decay,
                      target_reward="trajectories_td_prog.zarr")
  else:
    # print a warning
    print("[WARNING] Using old model!!")
    model = ModelBased(behavior_dataset.states.shape[1], #env.observation_spec().shape[0],
                        env.action_spec().shape[1], learning_rate=FLAGS.lr,
                        weight_decay=FLAGS.weight_decay)
    
  
  if FLAGS.load_mb_model:
    model.load("/app/ws/logdir/mb/mb_model_50000", "new_model")


  min_reward = tf.reduce_min(behavior_dataset.rewards)
  max_reward = tf.reduce_max(behavior_dataset.rewards)
  min_state = tf.reduce_min(behavior_dataset.states, 0)
  max_state = tf.reduce_max(behavior_dataset.states, 0)
  #print(min_state)
  #print(max_state)

  actor = F110Actor(FLAGS.target_policy, deterministic=False) #F110Stupid()
  model_input_normalizer = Normalize()

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
        #print("Scan 1")
        #print(laser_scan)
        laser_scan = model_input_normalizer.normalize_laser_scan(laser_scan)
        #print("Scan 2")
        #print(laser_scan)
      # back to dict
      model_input_dict = model_input_normalizer.unflatten_batch(batch_states_unnorm)
      # normalize back to model input
      model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
     
      # now also append the laser scan
      model_input_dict['lidar_occupancy'] = laser_scan
      #print("model input dict")
      #print(model_input_dict)

      batch_actions = actor(
        model_input_dict,
        std=FLAGS.target_policy_std)[1]
      
      actions_list.append(batch_actions)

    actions = tf.concat(actions_list, axis=0)
    actions = tf.convert_to_tensor(actions)
    return actions


  #@tf.function
  def update_step():
    import time

    (states, scans, actions, next_states, next_scans, rewards, masks, weights,
     log_prob, timesteps) = next(tf_dataset_iter)

    #if not(FLAGS.load_mb_model):
    model.update(states, actions, next_states, rewards, masks,
                  weights)

  gc.collect()

  for i in tqdm.tqdm(range(FLAGS.num_updates), desc='Running Training',  mininterval=5.0):
    #indices = np.where(behavior_dataset.mask_inital == 1)[0][:5]
    #print(indices)
    # print(tf.tensor(indices))
    #selected_states = tf.gather(behavior_dataset.states, indices)
    #print(selected_states)
    #print(behavior_dataset.initial_states[:5])
    update_step()

    if i % FLAGS.eval_interval == 0:
      horizon = 500
      print("Starting evaluation")
      if True:

        for j, evaluation_dataset in enumerate(eval_datasets):
          eval_ds = model.evaluate(evaluation_dataset.states,
                                  evaluation_dataset.actions,
                                  evaluation_dataset.rewards,
                                  evaluation_dataset.next_states,
                                  j,
                                  eval_agents[j],
                                  clip=True,
                                  min_reward=min_reward,
                                  max_reward=max_reward,
                                  min_state=min_state,
                                  max_state=max_state,)
        
      #pred_returns, std = model.estimate_returns(behavior_dataset.initial_states,
      #                       behavior_dataset.initial_weights,
      #                       get_target_actions, horizon=100,
      #                       discount=FLAGS.discount,)
      print("*returns*")
      #print(pred_returns)
      #print(std)
      #model.evaluate_rollouts(eval_datasets[0], behavior_dataset.unnormalize_rewards,
      #                        horizon=25, num_samples=100)
      # exit()
      evaluation_dataset_ = eval_datasets[0]
      model.plot_rollouts_fixed(evaluation_dataset_.states,
                    evaluation_dataset_.actions,
                    evaluation_dataset_.mask_inital,
                    min_state, max_state, 
                    horizon= 50,
                    num_samples=10,
                    path = f"logdir/plts/mb/mb_rollouts_{FLAGS.target_policy}_{FLAGS.discount}_{i}_0911.png",
                    get_target_action=get_target_actions)#np.max(behavior_dataset.steps) + 1)
      print("----")
      model.plot_rollouts_fixed(evaluation_dataset_.states,
              evaluation_dataset_.actions,
              evaluation_dataset_.mask_inital,
              min_state, max_state, 
              horizon= 50,
              num_samples=10,
              path = f"logdir/plts/mb/mb_rollouts_{FLAGS.target_policy}_{FLAGS.discount}_{i}_0911_fixed.png",
              get_target_action=None)#np.max(behavior_dataset.steps) + 1)


      model.save(f"/app/ws/logdir/mb/mb_model_{i}", "new_model")
      # print saved model
      print(f"saved model as /app/ws/logdir/mb/mb_model_{i}")

app.run(main)
if __name__ == '__main__':
  print("Running main")
  app.run(main)
