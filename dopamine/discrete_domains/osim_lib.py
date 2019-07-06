# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which
is in charge of:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


import gin
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf

from osim.env import L2M2019Env


slim = tf.contrib.slim


OSIM_DQN_OBSERVATION_SHAPE = (290,)  # Size of observation space
OSIM_DQN_OBSERVATION_DTYPE = tf.float32  # value of.
OSIM_DQN_STACK_SIZE = 4  # Number of frames in the state stack.
gin.constant('osim_lib.OSIM_DQN_OBSERVATION_SHAPE', (290,))
gin.constant('osim_lib.OSIM_DQN_OBSERVATION_DTYPE', tf.float32)
gin.constant('osim_lib.OSIM_DQN_STACK_SIZE', 4)


@gin.configurable
def create_opensim_environment(visualize=False, seed=None, difficulty=2, sticky_actions=True):
  print("Creating Environment .............")
  mode = '3D'
  env = L2M2019Env(visualize=visualize, seed=seed, difficulty=difficulty)
  env.change_model(model=mode, difficulty=difficulty, seed=seed)

  env = OpensimPreprocessing(env) 
  num_actions = 22
  env.action_space.n = num_actions 
  return env

@gin.configurable
def nature_dqn_network(num_actions, network_type, state):
  """The convolutional network used to compute the agent's Q-values.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """

  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  all_net = tf.cast(state, tf.float32)
  batch_size = all_net.get_shape().as_list()[0]
  stack_size = all_net.get_shape().as_list()[2]

  velocity_net = tf.slice(all_net, [0,0,0], [batch_size,242,stack_size])
  velocity_net = tf.reshape(velocity_net, [batch_size, 2, 11, 11, stack_size])

  velocity_net = slim.conv3d(
      velocity_net, 32, [1, 3, 3], stride=(1,2,2), weights_initializer=weights_initializer)
  velocity_net = slim.conv3d(
      velocity_net, 64, [1, 3, 3], stride=(1,2,2), weights_initializer=weights_initializer)      
  velocity_net = slim.conv3d(
      velocity_net, 64, [1, 3, 3], stride=(1,2,2), weights_initializer=weights_initializer)     
  velocity_net = slim.flatten(velocity_net)
  velocity_net = slim.fully_connected(velocity_net, 64, activation_fn=tf.nn.relu6)
  velocity_net = slim.fully_connected(velocity_net, 16, activation_fn=tf.nn.relu6)

  pelvis_net = tf.slice(all_net, [0,242,0], [batch_size,4,stack_size])
  pelvis_net = slim.flatten(pelvis_net)
  pelvis_net = slim.fully_connected(velocity_net, 4, activation_fn=tf.nn.relu6)

  l_leg = tf.slice(all_net, [0,246,0], [batch_size,22,stack_size])
  l_leg = slim.fully_connected(l_leg, 16, activation_fn=tf.nn.relu6)
  l_leg = slim.flatten(l_leg)
  l_leg = slim.fully_connected(l_leg, 8, activation_fn=tf.nn.relu6)

  r_leg = tf.slice(all_net, [0,268,0], [batch_size,22,stack_size])
  r_leg = slim.fully_connected(r_leg, 16, activation_fn=tf.nn.relu6)
  r_leg = slim.flatten(r_leg)
  r_leg = slim.fully_connected(r_leg, 8, activation_fn=tf.nn.relu6)

  body_net =  tf.concat([pelvis_net, l_leg], 1)
  body_net =  tf.concat([body_net, r_leg], 1)
  body_net = slim.flatten(body_net)
  body_net = slim.fully_connected(body_net, 64)

  net = tf.concat([pelvis_net, body_net], 1)
  net = slim.fully_connected(net, 128)
  net = slim.fully_connected(net, 64)

  q_values = slim.fully_connected(net, num_actions, activation_fn=tf.nn.relu)
  return network_type(q_values)

@gin.configurable
def rainbow_network(num_actions, num_atoms, support, network_type, state):
  """The convolutional network used to compute agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """

  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  all_net = tf.cast(state, tf.float32)
  batch_size = all_net.get_shape().as_list()[0]
  stack_size = all_net.get_shape().as_list()[2]

  velocity_net = tf.slice(all_net, [0,0,0], [batch_size,242,stack_size])
  velocity_net = tf.reshape(velocity_net, [batch_size, 2, 11, 11, stack_size])

  velocity_net = slim.conv3d(
      velocity_net, 32, [1, 3, 3], stride=(1,2,2), weights_initializer=weights_initializer)
  velocity_net = slim.conv3d(
      velocity_net, 64, [1, 3, 3], stride=(1,2,2), weights_initializer=weights_initializer)      
  velocity_net = slim.conv3d(
      velocity_net, 64, [1, 3, 3], stride=(1,2,2), weights_initializer=weights_initializer)     
  velocity_net = slim.flatten(velocity_net)
  velocity_net = slim.fully_connected(velocity_net, 32, activation_fn=tf.nn.relu6)

  pelvis_net = tf.slice(all_net, [0,242,0], [batch_size,4,stack_size])
  pelvis_net = slim.flatten(pelvis_net)
  pelvis_net = slim.fully_connected(velocity_net, 4, activation_fn=tf.nn.relu6)

  l_leg = tf.slice(all_net, [0,246,0], [batch_size,22,stack_size])
  l_leg = slim.fully_connected(l_leg, 16, activation_fn=tf.nn.relu6)
  l_leg = slim.flatten(l_leg)
  l_leg = slim.fully_connected(l_leg, 8, activation_fn=tf.nn.relu6)

  r_leg = tf.slice(all_net, [0,268,0], [batch_size,22,stack_size])
  r_leg = slim.fully_connected(r_leg, 16, activation_fn=tf.nn.relu6)
  r_leg = slim.flatten(r_leg)
  r_leg = slim.fully_connected(r_leg, 8, activation_fn=tf.nn.relu6)

  body_net =  tf.concat([pelvis_net, l_leg], 1)
  body_net =  tf.concat([body_net, r_leg], 1)
  body_net = slim.flatten(body_net)
  body_net = slim.fully_connected(body_net, 32)

  net = tf.concat([pelvis_net, body_net], 1)
  net = slim.fully_connected(net, 32)

  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)

@gin.configurable
def implicit_quantile_network(num_actions, quantile_embedding_dim,
                              network_type, state, num_quantiles):
  """The Implicit Quantile ConvNet.

  Args:
    num_actions: int, number of actions.
    quantile_embedding_dim: int, embedding dimension for the quantile input.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    num_quantiles: int, number of quantile inputs.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """

  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  all_net = tf.cast(state, tf.float32)
  batch_size = all_net.get_shape().as_list()[0]
  stack_size = all_net.get_shape().as_list()[2]

  velocity_net = tf.slice(all_net, [0,0,0], [batch_size,242,stack_size])
  velocity_net = tf.reshape(velocity_net, [batch_size, 2, 11, 11, stack_size])

  velocity_net = slim.conv3d(
      velocity_net, 32, [1, 3, 3], stride=(1,2,2), weights_initializer=weights_initializer)
  velocity_net = slim.conv3d(
      velocity_net, 64, [1, 3, 3], stride=(1,2,2), weights_initializer=weights_initializer)      
  velocity_net = slim.conv3d(
      velocity_net, 64, [1, 3, 3], stride=(1,2,2), weights_initializer=weights_initializer)     
  velocity_net = slim.flatten(velocity_net)
  velocity_net = slim.fully_connected(velocity_net, 64, activation_fn=tf.nn.relu6)
  velocity_net = slim.fully_connected(velocity_net, 16, activation_fn=tf.nn.relu6)

  pelvis_net = tf.slice(all_net, [0,242,0], [batch_size,4,stack_size])
  pelvis_net = slim.flatten(pelvis_net)
  pelvis_net = slim.fully_connected(velocity_net, 4, activation_fn=tf.nn.relu6)

  l_leg = tf.slice(all_net, [0,246,0], [batch_size,22,stack_size])
  l_leg = slim.fully_connected(l_leg, 16, activation_fn=tf.nn.relu6)
  l_leg = slim.flatten(l_leg)
  l_leg = slim.fully_connected(l_leg, 8, activation_fn=tf.nn.relu6)

  r_leg = tf.slice(all_net, [0,268,0], [batch_size,22,stack_size])
  r_leg = slim.fully_connected(r_leg, 16, activation_fn=tf.nn.relu6)
  r_leg = slim.flatten(r_leg)
  r_leg = slim.fully_connected(r_leg, 8, activation_fn=tf.nn.relu6)

  body_net =  tf.concat([pelvis_net, l_leg], 1)
  body_net =  tf.concat([body_net, r_leg], 1)
  body_net = slim.flatten(body_net)
  body_net = slim.fully_connected(body_net, 64)

  net = tf.concat([pelvis_net, body_net], 1)
  net = slim.fully_connected(net, 128)
  state_net = slim.fully_connected(net, 64)





  state_net_size = state_net.get_shape().as_list()[-1]
  state_net_tiled = tf.tile(state_net, [num_quantiles, 1])

  batch_size = state_net.get_shape().as_list()[0]
  quantiles_shape = [num_quantiles * batch_size, 1]
  quantiles = tf.random_uniform(
      quantiles_shape, minval=0, maxval=1, dtype=tf.float32)

  quantile_net = tf.tile(quantiles, [1, quantile_embedding_dim])
  pi = tf.constant(math.pi)
  quantile_net = tf.cast(tf.range(
      1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
  quantile_net = tf.cos(quantile_net)
  quantile_net = slim.fully_connected(quantile_net, state_net_size,
                                      weights_initializer=weights_initializer)
  # Hadamard product.
  net = tf.multiply(state_net_tiled, quantile_net)

  net = slim.fully_connected(
      net, 512, weights_initializer=weights_initializer)
  quantile_values = slim.fully_connected(
      net,
      num_actions,
      activation_fn=None,
      weights_initializer=weights_initializer)

  return network_type(quantile_values=quantile_values, quantiles=quantiles)


@gin.configurable
class OpensimPreprocessing(object):
  """A class implementing opensim preprocessing for Prosthetic agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

  def __init__(self, environment, frame_skip=2):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    
    self.environment = environment
    self.frame_skip = frame_skip

    #obs_dims = self.environment.observation_space


    #print("BIKS: obs_dims", obs_dims)

    # Stores temporary observations used for pooling over two successive
    # frames.
    #self.obs_buffer = [
    #    np.empty((obs_dims.shape[0]), dtype=np.float32),
    #    np.empty((obs_dims.shape[0]), dtype=np.float32)
    #]

    self.game_over = False

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0.0, high=1.0, shape=OSIM_DQN_OBSERVATION_SHAPE, dtype=np.float32)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def close(self):
    return self.environment.close()

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    INIT_POSE = np.array([
    1.699999999999999956e+00, # forward speed
    .5, # rightward speed
    9.023245653983965608e-01, # pelvis height
    2.012303881285582852e-01, # trunk lean
    0*np.pi/180, # [right] hip adduct
    -6.952390849304798115e-01, # hip flex
    -3.231075259785813891e-01, # knee extend
    1.709011708233401095e-01, # ankle flex
    0*np.pi/180, # [left] hip adduct
    -5.282323914341899296e-02, # hip flex
    -8.041966456860847323e-01, # knee extend
    -1.745329251994329478e-01]) # ankle flex

    sim_dt = 0.01
    sim_t = 10
    timstep_limit = int(round(sim_t/sim_dt))


    obs_dict = self.environment.reset(project=True, seed=None, obs_as_dict=True, init_pose=INIT_POSE)
    self.environment.spec.timestep_limit = timstep_limit
    return np.array(self._obs_dict_to_arr(obs_dict))
    #return self._pool_and_resize()

  def _obs_dict_to_arr(self, obs_dict):
          # Augmented environment from the L2R challenge
    res = []

    # target velocity field (in body frame)
    v_tgt = np.ndarray.flatten(obs_dict['v_tgt_field'])
    res += v_tgt.tolist()

    res.append(obs_dict['pelvis']['height'])
    #res.append(obs_dict['pelvis']['pitch'])
    #res.append(obs_dict['pelvis']['roll'])
    res.append(obs_dict['pelvis']['vel'][0]/self.environment.LENGTH0)
    res.append(obs_dict['pelvis']['vel'][1]/self.environment.LENGTH0)
    res.append(obs_dict['pelvis']['vel'][2]/self.environment.LENGTH0)
    #res.append(obs_dict['pelvis']['vel'][3])
    #res.append(obs_dict['pelvis']['vel'][4])
    #res.append(obs_dict['pelvis']['vel'][5])

    for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(obs_dict[leg][MUS]['f'])
                #res.append(obs_dict[leg][MUS]['l'])
                #res.append(obs_dict[leg][MUS]['v'])
    return res

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action):
    accumulated_reward = 0.0


    #Sticky Action while skipping observation
    specifix_action = np.zeros(self.environment.action_space.n, dtype=np.float32)
    specifix_action[action] = 1.0

    for time_step in range(self.frame_skip):
      # We bypass the Gym observation altogether and directly fetch the
      # grayscale image from the ALE. This is a little faster.


      #print('BIKS: ACTION: ', action)


      obs_dict, reward, done, info = self.environment.step(specifix_action, project = True, obs_as_dict=True)
      accumulated_reward += reward

      if done:
        break
      # We max-pool over the last two observation.
      #elif time_step >= self.frame_skip - 2:
      #  t = time_step - (self.frame_skip - 2)
      #  self._fetch_grayscale_observation(self.screen_buffer[t])

    # Pool the last two observations.
    #observation = self._pool_and_resize()
    
    observation = np.array(self._obs_dict_to_arr(obs_dict))

    self.game_over = done
    return observation, accumulated_reward, done, info

