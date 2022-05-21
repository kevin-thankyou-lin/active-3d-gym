import gym
import active_vision

env = gym.make('OfflineActiveVision-v0', data_dir="/home/guest/code/active-nerf/output")
env.reset()
env.step(1)