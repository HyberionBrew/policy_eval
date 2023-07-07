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

"""Behavior cloning learning."""
import tensorflow.compat.v2 as tf
from tensorflow_addons import optimizers as tfa_optimizers
from tf_agents.specs import tensor_spec
import fabian.ws_ope.policy_eval.actor_copy as actor_lib

import numpy as np

class BehaviorCloning(object):
  """Behavior cloning."""

  def __init__(self,
               state_dim,
               action_spec,
               learning_rate,
               weight_decay):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      learning_rate: Learning rate.
      weight_decay: Weight decay.
    """
    self.actor = actor_lib.Actor(state_dim, action_spec)
    self.action_spec = action_spec
    self.optimizer = tfa_optimizers.AdamW(learning_rate=learning_rate,
                                          weight_decay=weight_decay)

  def __call__(self, states, actions, batch_size=4000):
    # time this call 
    import time
    start = time.time()

    with tf.device('CPU:0'):
      dist, _ = self.actor.get_dist_and_mode(states)
      #print("*********")
      #print(actions.shape)
      #print(dist.shape)
      # do this in batches to avoid OOM
    
      log_probs = dist.log_prob(actions)
    # print("Time for call: ", time.time() - start)
    """
    num_batches = int(np.ceil(len(actions) / batch_size))
    print(num_batches)
    print(len(actions))
    print("-------------")
    # Placeholder for collecting log_probs from all batches
    log_probs_list = []
    for i in range(num_batches):
      # Calculate start and end indices for the current batch
      start_idx = i * batch_size
      end_idx = min((i + 1) * batch_size, len(actions))
      
      # Extract the current batch of actions
      batch_actions = actions[start_idx:end_idx]

      # Clip actions
      batch_actions = tf.clip_by_value(batch_actions, 1e-4 + self.action_spec.low,
                                        -1e-4 + self.action_spec.high)
      
      # Compute log_probs for the current batch
      print(batch_actions.shape)
      batch_log_probs = dist.log_prob(batch_actions)
      
      # Collect the batch_log_probs
      log_probs_list.append(batch_log_probs)
    # Concatenate the collected log_probs from all batches
    log_probs = tf.concat(log_probs_list, axis=0)
    """
    return dist, log_probs

  @tf.function
  def update(self,
             states,
             actions,
             weights):
    """Updates actor parameters.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      weights: A batch of weights.
    Returns:
      Actor loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.trainable_variables)
      actions = tf.clip_by_value(actions, 1e-4 + self.action_spec.low,
                                 -1e-4 + self.action_spec.high)
      log_prob = self.actor.get_log_prob(states, actions)
      actor_loss = (
          tf.reduce_sum(-log_prob * weights) /
          tf.reduce_sum(weights))
    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))
    # only log every 1000 steps
    if self.optimizer.iterations % 1000 == 0:
      tf.summary.scalar('train/actor loss', actor_loss,
                        step=self.optimizer.iterations)
