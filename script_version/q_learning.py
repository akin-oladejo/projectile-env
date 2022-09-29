from gym import Env
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import sleep
import seaborn as sns

class Q():
    def __init__(self, 
                 env:Env,
                 min_epsilon:float = 0.02,
                 max_epsilon:float = 0.5,
                 exploration_fraction:float = 0.4,
                 lr:float = 0.003,
                 gamma:float = 1.0):
        
        # store the environment
        self.env = env

        # create table
        self.qtable = np.zeros(shape=(self.env.n_states, self.env.n_actions))

        # set hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.exploration_fraction = exploration_fraction
        
       

    def greedy_policy(self):
        '''
        Choose the action with the highest q-value for that state
        '''
        return np.argmax(self.qtable[self.state])


    def epsilon_greedy_policy(self, epsilon):
        '''
        Choose any action sometimes, other times choose the one with high q-value
        '''
        # generate probability value to match epsilon against
        rando = random.uniform(0,1)

        # if value < epsilon, explore (because we will be decaying epsilon gradually)
        if rando < epsilon:
            action = self.env.action_space.sample()
        # else, exploit
        else:
            action = np.argmax(self.qtable[self.state])
        
        return action

    def train(self, train_episodes:int = 5000): 
        
        # calculate the number of timesteps to anneal exploration
        self.n_exploration_timesteps = int(self.exploration_fraction * train_episodes )
        self.epsilon_values = np.linspace(self.max_epsilon, self.min_epsilon, self.n_exploration_timesteps)

        for episode in tqdm(range(train_episodes)):
            # adjust epsilon
            # self.epsilon = self.min_epsilon + (self.max_epsilon-self.min_epsilon)*np.exp(-self.decay_rate*episode)
            if episode < self.n_exploration_timesteps:
                self.epsilon = self.epsilon_values[episode]
            else:
                self.epsilon = self.min_epsilon

            self.state, _ = self.env.reset(verbose=False) # reset the episode
            
            #train loop
            while not self.env.done:
                # print(self.state)
                if episode < self.n_exploration_timesteps:
                    self.action = self.env.action_space.sample()
                else:
                    self.action = self.greedy_policy()

                # take an ε-greedy action
                # self.action = self.epsilon_greedy_policy(self.epsilon) # uncomment this
                self.new_state, reward, done, info = self.env.step(self.action)
                
                # update self.qtable using Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.td_target = reward + self.gamma*np.max(self.qtable[self.new_state]) # compute the one-step return
                self.td_error = self.td_target - self.qtable[self.state][self.action]
                self.qtable[self.state][self.action] = self.qtable[self.state][self.action] + (self.lr * self.td_error)
  
                # # Render the env
                # self.env.render()
                # # Wait a bit before the next frame 
                # sleep(0.001)

                # if done, terminate
                if done:
                    break

                self.state = self.new_state


    def __repr__(self):
        return f"Run the 'Q.show()' method to show the Qtable better\n{self.qtable}"
        
    def show(self, save_as:str):
        # plot the qtable as a heatmap
        plt.figure(figsize=(10,5))
        plt.title(f'Qtable for {self.env.name} environment')
        ax = sns.heatmap(self.qtable, annot=True, linecolor='pink', linewidths=0.2, cmap='Greys')
        ax.set(xlabel='angle(°)', ylabel='target distance(m)') # label axes
        plt.xticks(ticks=np.arange(self.env.n_actions)+0.5, labels=self.env.action_to_angle.values(), ha='center') # change xticks to angles
        state_to_target = {j:i for i,j in self.env.target_to_state.items()}
        plt.yticks(ticks=np.arange(self.env.n_states)+0.5, labels=state_to_target.values(), va='center')
        
        # save image to file
        if save_as:
            plt.savefig(save_as)