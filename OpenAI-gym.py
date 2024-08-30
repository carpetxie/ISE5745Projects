!pip install gym==0.22
!apt-get install python-opengl -y
!apt install xvfb -y

# Special gym environment
!pip install gym[atari]

# For rendering environment, you can use pyvirtualdisplay.
!pip install pyvirtualdisplay
!pip install piglet

# To activate virtual display
# need to run a script once for training an agent as follows
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

# This code creates a virtual display to draw game images on.
# If you are running locally, just ignore it
import os
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
    !bash ../xvfb start
    %env DISPLAY=:1

#
# Import libraries
#
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) # error only
#import tensorflow as tf
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import math
import glob
import io
import base64
from IPython.display import HTML

from IPython import display as ipythondisplay

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else:
    print("Could not find video")


def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

# I have included code to help you discretize the state space.
# You DO NOT need to keep these specific bin ranges.
# In fact, you may not want to keep these bin ranges.
# I have provided this code to make it easier for you to modify and to save
# you time.
# You can alter or discretize the state space however you wish.
# You do not need to keep all 4 state features if you have an argument for
# eliminating features.

import pandas as pd

# Discretize input state to make Q-table and to reduce dimensionality
def discretize(state):

  #print ( state )

  # First, set up arrays of the left bin edges
  # Note: your bin sizes do not need to be of uniform width.
  bins_pos = np.linspace(-2.4, 2.4, 20)
  bins_vel = np.linspace(-4,4, 20)
  bins_ang = np.linspace(-0.2095, 0.2095, 20)
  bins_w = np.linspace(-4,4, 20)
  #rememebr to use linspace for segmentations

  cart_position_bin = np.digitize(state[0], bins=bins_pos) - 1
  cart_velocity_bin = np.digitize(state[1], bins=bins_vel) - 1
  pole_angle_bin = np.digitize(state[2], bins=bins_ang) - 1
  angle_rate_bin = np.digitize(state[3], bins=bins_w) - 1
  #indexing

  # To verify the order of the state variables:
  #   https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

  return [cart_position_bin, cart_velocity_bin, pole_angle_bin, angle_rate_bin]

# Simple code to test your discretization.

# Specify which environment to use.

env = gym.make('CartPole-v0')
state = env.reset()

action = env.action_space.sample() # Explore action space
state, reward, done, info = env.step(action)
print ( 'Continuous state: ')
print ( state )

discretized_state = discretize(state)
print ( 'Discretized state: ')
print ( discretized_state )

env.close()

# Specify which environment to use.
env = gym.make('CartPole-v0')
env = wrap_env(env)
#state = env.reset()

np.random.seed(42)
##########################################
# Initialize your Q-values.
# Note: you may use whichever data structure you wish.
#       I used a dictionary, but a list works, too.
##########################################

q_table = {}

##########################################
# Initialize RL Parameters
##########################################

epsilon = 0.95
epsilon_min = 0.01

  #0.95 * x^2000 = 0.01
  #x = 0.9977

epsilon_decay_rate = 0.9988

learning_rate = 0.09

discount_factor = 0.99

# For plotting metrics
cumulative_reward_each_episode = []
epsilon_each_episode = []


# To start off wish debugging your code, use 1 episode. Increase this once
# your code starts to work.
maxNumEpisodes = 4000

# For each episode
for i in range(maxNumEpisodes):

  # Reset to initial conditions
  state = env.reset()
  ##########################################
  # Discretize the state
  # Note: you'll need to modify the discretize function
  #       provided above.
  ##########################################
  state = discretize(state)


	# At the beginning of each episode, set the cumulative reward variable to zero.
  cumulative_reward = 0
  done = False

  # For every step in the episode
  while not done:
    state = tuple(state)
    env.render()
    if np.random.rand() < epsilon:
      action = env.action_space.sample()
    else:
      q_values = []
      for action_v in (0,1):
        q_values.append(q_table.get((*state, action_v), 0))
      action = np.argmax(q_values)


    ##########################################
    # For every time step, using epsilon-greedy to choose between
    # exploration and exploitation.
    # Implement epsilon-greedy exploration.
    # Hint: to return a random action, do this:
    #           action = env.action_space.sample()
    ##########################################



		# Take the action.
		# This moves the agent to a new state and earns a reward
    next_state, reward, done, info = env.step(action)
    # Discrete the state
    next_state = discretize(next_state)
    next_state = tuple(next_state)

    # Add the reward just earned to the cumulative reward variable
    cumulative_reward += reward

		##########################################
    # Update your estimate of Q(s,a)
    ##########################################

    next_max_values = []
    for a in (0, 1):
        next_max_values.append(q_table.get((*next_state, a), 0))
    next_max_q_value = max(next_max_values)

    #print("state,action", q_table.get((*state, action), 0))
    #print("reward", reward)
    #print("next_max_q_value", next_max_q_value)

    q_table[(*state, action)] = (1 - learning_rate) * q_table.get((*state, action), 0) + learning_rate * (reward + discount_factor * next_max_q_value)
    #0.11 * (1+0.9*0)






    state = next_state

    # If the episode is finished, do a few things.
    if done:
      # Save the cumulative reward from the previous episode to an array.
      cumulative_reward_each_episode.append(cumulative_reward)

      # Save the epsilon used in this episode.
      epsilon_each_episode.append(epsilon)

      ##########################################
      # Decay epsilon.
      # If you want to decay or change the value of epsilon at the end of
      # each episode, do so here.
      epsilon *= epsilon_decay_rate
      ##########################################


  if i % 100 == 0:
    print('Episode: {0}'.format(i))
env.close()
show_video()
print("Training finished.\n")


# Plot the Cumulative Reward and Epsilon value through time.
fsize = 12

plt.plot(cumulative_reward_each_episode)
plt.title('Cumulative Reward through Time', fontsize=fsize)
plt.xlabel('Episode', fontsize=fsize)
plt.ylabel('Cumulative Reward', fontsize=fsize)
plt.show()


plt.plot(epsilon_each_episode)
plt.title('Exploration (epsilon) through Time', fontsize=fsize)
plt.xlabel('Episode', fontsize=fsize)
plt.ylabel('epsilon', fontsize=fsize)
plt.show()

