import numpy as np
import random


def epsilon_greedy_action(env, Q, state, epsilon):
    action = env.action_space.sample()  # Explore action space
    
    
    # with probability 1-eps we go greedy, 
    # else we remain random
    if random.random() < (1-epsilon):
        action = np.argmax(Q[state])
        
    return action


def sarsa_lambda(env, alpha=0.2, gamma=0.99, lambda_= 0.3, initial_epsilon=1.0, n_episodes=30000 ):

    ####### Hyperparameters
    # alpha = learning rate
    # gamma = discount factor
    # lambda_ = elegibility trace decay
    # initial_epsilon = initial epsilon value
    # n_episodes = number of episodes

    ############# keep this shape for the Q!
    Q = np.random.rand(env.observation_space.n, env.action_space.n)

    # init epsilon
    epsilon = initial_epsilon

    received_first_reward = False

    print("TRAINING STARTED")
    print("...")
    for ep in range(n_episodes):
        state, _ = env.reset()
        action = epsilon_greedy_action(env, Q, state, epsilon)
        done = False
        
        # reset the eligibility
        E = np.zeros_like(Q)
        
        while not done:
            ############## simulate the action
            next_state, reward, done, info, _ = env.step(action)
            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)


            ############## update q table and eligibility
            delta = reward + gamma*Q[next_state, next_action] - Q[state, action]
            E[state, action] +=  1
            
            # update for every s and a
            Q = Q + alpha*delta*E
            E = gamma*lambda_*E

            if not received_first_reward and reward > 0:
                received_first_reward = True
                print("Received first reward at episode ", ep)

            # update current state and action
            state = next_state
            action = next_action

        # update current epsilon
        if received_first_reward:
            epsilon = 0.999 * epsilon
            
    print("TRAINING FINISHED")
    return Q