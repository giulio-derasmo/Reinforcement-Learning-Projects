import random

import numpy as np
import gym
import time
from gym import spaces
import os

def value_iteration(env):
    gamma=0.99
    iters=100

    #initialize values
    values = np.zeros((env.num_states))
    best_actions = np.zeros((env.num_states), dtype=int)
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1

    for i in range(iters):
        v_old = values.copy()
        for s in range(env.num_states):
            state = STATES[s]

            if (state == env.end_state).all() or i >= env.max_steps:
                continue # if we reach the termination condition, we cannot perform any action
                
            max_va = -np.inf
            best_a = 0
            for a in range(env.num_actions):
                next_state_prob = env.transition_probabilities(state, a).flatten()
                
                va = (next_state_prob*(REWARDS + gamma*v_old)).sum()

                if va > max_va:
                    max_va = va
                    best_a = a
            values[s] = max_va
            best_actions[s] = best_a

    return best_actions.reshape((env.height, env.width))


def policy_iteration(env, gamma=0.99, iters=100):
    policy = np.zeros(env.num_states, dtype=np.int)
   
    
    #initialize values
    values = np.zeros((env.num_states))
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()
    
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1
    
    
    # Policy iteration 
    policy_stable = False
    for j in range(iters): 
        
        # termination condition if the policy doesn't improve more
        # or exceed number of max iteration
        if policy_stable or j >= env.max_steps:
            continue
        
        ## Iterative Policy Evaluation 
        for i in range(iters):
            v_old = values.copy()
            # loop for each s in S
            for s in range(env.num_states):
                state = STATES[s]
    
                if (state == env.end_state).all() or i >= env.max_steps:
                    continue # if we reach the termination condition, we cannot perform any action
      
                # compute the value function according to the current policy
                next_state_prob = env.transition_probabilities(state, policy[s]).flatten()
                values[s] = (next_state_prob*(REWARDS + gamma*v_old)).sum()
        
    
        ## Policy improvement 
        for s in range(env.num_states):
            state = STATES[s]
            old_action = policy[s]
            
            max_va = -np.inf
            for a in range(env.num_actions):
                next_state_prob = env.transition_probabilities(state, a).flatten()
                va = (next_state_prob*(REWARDS + gamma*values)).sum()
                if va > max_va:
                    max_va = va
                    # update policy 
                    policy[s] = a
    
            if policy[s] != old_action:
                policy_stable = False

    
    return policy.reshape(env.height, env.width)