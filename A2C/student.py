import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms 

class ActorNetwork(nn.Module):
    """
    Class for the ActorNetwork: used to learn the policy pi( |s)
    """
    def __init__(self, action_size, batch_size):
        super(ActorNetwork, self).__init__()
        """select and initialize the Device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        Build the neural networks
        according to https://web.stanford.edu/class/aa228/reports/2019/final9.pdf
        """
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 8, stride = 4)
        self.do1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2)
        self.do2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1)
        self.do3 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(128*6*6, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, action_size)
        self.flatten = torch.nn.Flatten(1, -1)
        "initialize gray_scale transformer"
        self.gs = transforms.Grayscale()
        "initialize batch size"
        self.batch_size = batch_size
        self.to(self.device)

    def preprocess_state(self, state):
        """
        Function in order to preprocess the state
        """
        # remove black under image
        state = state[:83, :83] 
        # make the car black
        state[67:77, 42:53] = 0
        # unify color grass
        # light_grass = [100,228,100]
        # dark_grass  = [100,202,100]
        state = np.where(state == [100,228,100], [100,202,100], state)
        # unify street color 
        state[(state >= 100) & (state <= 105)] = 100
        # to gray scale
        state = self.gs(torch.from_numpy(state).permute(2,0,1))
        # normalize 
        state = state / 255.
        return state

    def forward(self, x):
        """Apply pre process"""
        if x.shape[0] != self.batch_size:
            x = self.preprocess_state(x).unsqueeze(0)
        else: 
            x = torch.vstack([self.preprocess_state(state).unsqueeze(0) for state in x])
        """Forward step"""
        x = x.float().to(self.device)
        x = F.relu(self.conv1(x))
        x = self.do1(x)
        x = F.relu(self.conv2(x))
        x = self.do2(x)
        x = F.relu(self.conv3(x))
        x = self.do3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        if x.shape[0] == self.batch_size:
            x = self.bn1(x)
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    
    def get_action(self, x):
        """Function that return the action to perform"""
        m = torch.distributions.Categorical(self.forward(x))
        action = m.sample().cpu().detach().numpy()
        return action
    
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

class ValueNetwork(nn.Module):
    """Class for the ValueNetwork: used to learn the critic V(s)"""
    def __init__(self, output_size, batch_size):
        super(ValueNetwork, self).__init__()
        """select and initialize the Device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        Build the neural networks
        according to https://web.stanford.edu/class/aa228/reports/2019/final9.pdf
        """
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 8, stride = 4)
        self.do1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2)
        self.do2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1)
        self.do3 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(128*6*6, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, output_size)
        self.flatten = torch.nn.Flatten(1, -1)
        "initialize gray_scale transformer"
        self.gs = transforms.Grayscale()
        "initialize batch size"
        self.batch_size = batch_size
        self.to(self.device)
        

    def preprocess_state(self, state):
        """
        Function in order to preprocess the state
        """
        # remove black under image
        state = state[:83, :83] 
        # make the car black
        state[67:77, 42:53] = 0
        # unify color grass
        # light_grass = [100,228,100]
        # dark_grass  = [100,202,100]
        state = np.where(state == [100,228,100], [100,202,100], state)
        # unify street color 
        state[(state >= 100) & (state <= 105)] = 100
        # to gray scale
        state = self.gs(torch.from_numpy(state).permute(2,0,1))
        # normalize 
        state = state / 255.
        return state

    def forward(self, x):
        """Apply pre process"""
        if x.shape[0] != self.batch_size:
            x = self.preprocess_state(x).unsqueeze(0)
        else: 
            x = torch.vstack([self.preprocess_state(state).unsqueeze(0) for state in x])
        """Forward step"""
        x = x.float().to(self.device)
        x = F.relu(self.conv1(x))
        x = self.do1(x)
        x = F.relu(self.conv2(x))
        x = self.do2(x)
        x = F.relu(self.conv3(x))
        x = self.do3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        if x.shape[0] == self.batch_size:
            x = self.bn1(x)
        x = self.fc2(x)
        return x

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret



class Policy:
    """
    Class for the agent
    Since trained on Colab some part are commented in order to have in student.py only
    the actual training part w/out testing
    """
    continuous = False 

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        """select and initialize the Device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """
        Initialize the enviroment for training and for testing
        > the env_eval is masked for the students.py
        """
        self.env = gym.make('CarRacing-v2', continuous = False)
        #self.env_eval = gym.make("CarRacing-v2", continuous = False, domain_randomize=False)
        self.gamma = 0.99
        self.n_epochs = 1   # 1 epoch to ensure everything works
        self.batch_size = 256

        """Initialize the networks with the relative optimizer"""
        self.value_network = ValueNetwork(1, self.batch_size)
        self.actor_network = ActorNetwork(self.env.action_space.n, self.batch_size)
        self.lr = 0.0001
        self.value_network_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.lr)
        self.actor_network_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=self.lr)
        
    def act(self, state):
        """Function to perferom the action in order to be used during the evaluation in main.py"""
        return self.actor_network.get_action(state).item()

    def compute_TD_Adv(self, rewards, dones, values, next_value):
        """
        Function to compute the n-step return and Advantage
        n-step G = \sum_{i=1}^n gamma^i * r_i + gamma^(n+1)*V(s_n+1)
        """
        # initialize the return array appending the last value in the end 
        # to loopy the return computation
        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)

        for t in reversed(range(rewards.shape[0])):
            # if S is terminal --> V(S) = 0. 
            # (1-done) treat this
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        
        # remove <next_value>
        returns = returns[:-1]  
        # A = G - V
        advantages = returns - values
        return returns, advantages

    def optim_step(self, observations, actions, returns, advantages):
        """Function to perform the optimation step"""

        # one-hot-vector in order to obtain the log(pi(a|s))
        # without having to access to distr.log_prob 
        actions = F.one_hot(torch.tensor(actions), self.env.action_space.n).to(self.device)
        # cast to tensor 
        returns = torch.tensor(returns[:, None], dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)

        # the value is optimized using the MSE loss
        self.value_network_optimizer.zero_grad()
        values = self.value_network.forward(observations).to(self.device)
        # MSE = (N-step G - V)^2
        loss_value = 1 * F.mse_loss(values, returns)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1)
        self.value_network_optimizer.step()

        # Actor loss
        self.actor_network_optimizer.zero_grad()
        probs = self.actor_network.forward(observations).to(self.device)
        # loss = log(pi(a|s))*advantage
        loss_policy = ((actions.float() * probs.log()).sum(-1) * advantages).mean()
        # entropy = pi(a|s)*log(pi(a|s))
        loss_entropy = - (probs * probs.log()).sum(-1).mean()
        # train_loss = - loss_policy - c2 * loss_entropy
        loss_actor = - loss_policy - 0.00001 * loss_entropy
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1)
        self.actor_network_optimizer.step()
        
        return loss_value, loss_actor

    def train(self):
        """Function to perform the train"""

        # initialize the object we want to store during the episodes
        actions = np.zeros((self.batch_size,), dtype=np.int64)
        dones = np.zeros((self.batch_size,), dtype=bool)
        rewards, values = np.zeros((2, self.batch_size), dtype=float)
        observations = np.zeros((self.batch_size,) + self.env.observation_space.shape, dtype=float)

        #rewards_test = []

        # initialize the episode with S_0
        observation = self.env.reset() 

        # loop
        for epoch in range(self.n_epochs):
            
            # the initial frames are rumor 
            # we skip that 
            for i in range(60):
                observation,_,_,_,_ = self.env.step(0)

            negative_patience = 50
            # loop for collecting the batch
            for i in range(self.batch_size):
                # storing step
                observations[i] = observation
                values[i]  = self.value_network.forward(observation).cpu().detach().numpy()
                actions[i] = self.actor_network.get_action(observation)
                observation, rewards[i], dones[i], _ , _ = self.env.step(actions[i])
                # help the agent to take the action "gas" and guide faster
                if actions[i] == 3 and rewards[i] > 0:
                    rewards[i] = 3
                # the negative patience is used to reward with an high negative value
                # the agent is it steps too much in the grass (not in the street)
                if rewards[i] < 0:
                    negative_patience -= 1
                else:
                    negative_patience = 50

                if negative_patience == 0:
                    rewards = -100 * np.ones_like(rewards)
                    dones[i] = True

                # if the episode finish, we reset the enviroment 
                if dones[i]:
                    observation, _ = self.env.reset()

            # If we finish the episode, V(s) = 0
            # if we just finish the n-step than we need to compute V(s+1)
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network.forward(observation).cpu().detach().numpy().item()

            # Compute returns and advantages
            returns, advantages = self.compute_TD_Adv(rewards, dones, values, next_value)
            # update the weights
            self.optim_step(observations, actions, returns, advantages)

            # Test it every 5 epochs and show the result
            # used only on colab, not required in students.py
            #if epoch % 5 == 0 or epoch == self.n_epochs - 1:
            #    rewards_test.append(np.array([self.evaluate() for _ in range(1)]))
            #    print(f'Epoch {epoch}/{self.n_epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')
            #    observation, _ = self.env.reset()
                    
        return rewards


   #def evaluate(self, render=False):
   #    env = self.env_eval
   #    reward_episode = 0
   #    done = False

   #    s, _ = env.reset()
   #    max_steps = 500
   #    for i in range(max_steps):
   #        action = self.act(s)
   #        s, reward, done, truncated, info = env.step(action)
   #        if render: env.render()
   #        reward_episode += reward
   #        if done or truncated: break
   #       
   #    env.close()
   #    if render:
   #        print(f'Reward: {reward_episode}')
   #    return reward_episode

    def save(self):
        """Function for saving the weights of both the newtorks"""
        torch.save({
            'actor_state_dict': self.actor_network.state_dict(),
            'critic_state_dict': self.value_network.state_dict(),
            }, 'model.pt')

    def load(self):
        """Load the weights in the class"""
        models_state_dict = torch.load('model.pt', map_location=torch.device('cpu'))
        self.actor_network.load_state_dict(models_state_dict['actor_state_dict'])
        self.value_network.load_state_dict(models_state_dict['critic_state_dict'])

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret