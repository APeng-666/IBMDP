'''
Wrap the environment to be iterative bounding, rewrite the following:
    1, observation
    2, action
    3, step()
    4, reward
    5, reset()
Current version considers only finite action and discretized bounding features, for box2d of openai gym
'''
import gym
from gym import spaces
import numpy as np
import random

class IBEnvDisWrapper():
    def __init__(self, env, discretization, bound, structure_reward, seed):
        np.random.seed(seed)
        self.seed = seed 

        self.env = env
        self.env.seed(self.seed)
        self.p = discretization 
        self.bound = bound    # bound for the inf features
        self.structure_reward = structure_reward # reward for constructing each level of the tree
        
        self.aug_state = 0
        self.features_dim = env.observation_space.shape[0]
        self.features_low = np.clip(env.observation_space.low, -self.bound, self.bound)
        self.features_high = np.clip(env.observation_space.high, -self.bound, self.bound)
        self.low = np.concatenate((self.features_low, self.features_low, self.features_low))
        self.high = np.concatenate((self.features_high, self.features_high, self.features_high))
        self.observation_space = spaces.Box(low = self.low, high = self.high)

        self.base_action_n = env.action_space.n
        self.action_to_id = dict()
        self.separation_points = np.array(list(range(1, self.p + 1))) / (self.p +1)
        index = 0
        for i in range(self.base_action_n):
            self.action_to_id[(i,0)] = index 
            index +=1
        for i in range(self.features_dim):
            for j in self.separation_points.tolist():
                self.action_to_id[(i,j)] = index 
                index += 1 

        self.id_to_action = list(self.action_to_id.keys())
        self.action_space = spaces.Discrete(self.base_action_n + self.p * self.features_dim)

    def step(self, action): # action is an integer
        act_i, act_j = self.id_to_action[action]
        if act_j:
            unnorm_feature = act_j * (self.aug_state[2 * self.features_dim + act_i] - self.aug_state[self.features_dim + act_i]) + self.aug_state[self.features_dim + act_i]
            if self.aug_state[act_i] <= unnorm_feature:
                self.aug_state[2*self.features_dim + act_i] = min(self.aug_state[2 * self.features_dim + act_i], unnorm_feature)
            else:
                self.aug_state[self.features_dim + act_i] = max(self.aug_state[self.features_dim + act_i], unnorm_feature)
            reward = self.structure_reward
            done = False
            if self.aug_state[self.features_dim + act_i] >= self.aug_state[2*self.features_dim + act_i]: # if lower >= upper
                done = True
                reward = - np.abs(self.structure_reward) * 10
                print('\nIterative bounding collapses', end="")
            
            info = {'traverse_tree': True, 'choose_base_action':False}
        else:
            next_state, reward, done, _ = self.env.step(act_i)
            info = {'traverse_tree': False, 'choose_base_action':True}
            self.aug_state = np.concatenate((next_state, self.features_low, self.features_high))
        return self.aug_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.aug_state = np.concatenate((state, self.features_low, self.features_high))
        return self.aug_state

