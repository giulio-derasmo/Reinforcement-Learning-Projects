import random
import numpy as np
import gym
import time
from gym import spaces
import os
import pickle


class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:
    def __init__(self, env): # modify
        self.env = env
        # create the rbf encoder
        self.num_rbf = 49
        self.rbf_sigma = 0.1
        self.rbf_den = 2 * self.rbf_sigma ** 2
        # divide the observation space (normalized) in grid of center
        self.cx, self.cy = np.meshgrid(np.linspace(0., 1., num=7),
                                       np.linspace(0., 1., num=7))
        
        self.centres = np.stack([self.cx.ravel(), self.cy.ravel()], axis = 1)
        
    def encode(self, state): # modify
        # encoding function
        
        # normalize the state
        box_space = self.env.observation_space
        state = 1 - (state - box_space.low) / (box_space.high - box_space.low)
        # compute the rbf expansion
        state = np.exp(-np.linalg.norm(state - self.centres, axis = 1) ** 2 / self.rbf_den)
            
        return state
    
    @property
    def size(self): # modify
        # return the correct size of the observation
        return self.num_rbf

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder,
        alpha=0.01, alpha_decay=1, 
        gamma=0.9999, epsilon=0.3, epsilon_decay=1, lambda_=0.9): # modify if you want
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        
    def Q(self, feats):
        feats = feats.reshape(-1,1)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)
                
        delta = reward 
        if not done:
            delta += self.gamma*self.Q(s_prime_feats).max()
        delta -= self.Q(s_feats)[action]
        
        # update weights and traces 
        self.traces[action] = self.gamma*self.lambda_*self.traces[action] + s_feats
        self.weights[action] += self.alpha * delta * self.traces[action]
        
    def update_alpha_epsilon(self): # modify
        self.epsilon *= self.epsilon_decay
        self.alpha *= self.alpha_decay
        
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None):  # modify
        if epsilon is None: epsilon = self.epsilon
        
        # with probability epsilon we explore
        # otherwise we go greedy
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.policy(state)

        return action
       
        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)
                
                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
