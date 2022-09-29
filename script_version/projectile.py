from gym import Env
import numpy as np
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
import random

class Projectile(Env):
    def __init__(self, initial_velocity=15, g=9.8, target_low=3, target_high=22, max_repeat=9):
        self.name = 'Projectile'
        # define environment bounds
        self.observation_space = Box(low=target_low, high=target_high, shape=(1,), dtype=np.int32)

        # there are 9 actions corresponding to the interval of angles from 5 to 45
        self.action_space = Discrete(9)

        # store these for easy access later
        self.n_actions = self.action_space.n
        self.n_states = target_high - target_low + 1

        self.v = initial_velocity # the default value for initial_velocity is 15m/s
        self.g = g # the default value for acceleration due to gravity is 9.8m/s²
        
        # the number of times to repeat a target value before moving on to the next
        self.max_repeat = max_repeat 

        # lookup the angle values for each action
        self.action_to_angle = {
            0:5,
            1:10,
            2:15,
            3:20,
            4:25, 
            5:30, 
            6:35, 
            7:40,
            8:45
        }

        # create a lookup of discrete state values corresponding to target distances
        self.target_to_state = {i:x for x,i in enumerate((range(target_low, target_high+1)))}

        # keep track of the lowest error attained for each target
        self.min_error = {x:np.inf for x in range(self.observation_space.low[0], self.observation_space.high[0] + 1)} 


    def new_target(self):
        # pick a target from the list of targets and remove it 
        self.target = self.target_list.pop()
        

    def get_obs(self, verbose=False):
        # self.target = self.observation_space.sample() # generate random target
        self.state = self.target_to_state[self.target] # represent target as state

        # if verbose=True, return the true value of the target distance, else return 
        # its representation as a state value
        if verbose:
            return f'Target distance: {self.target}m'
        else:
            return self.state


    def get_info(self, verbose=True):
        # return the number of meters by which we've missed the target
        self.error = np.abs(self.target - self.range) # absolute error

        # if verbose=True, return a full sentence, else just return the value of the error
        if verbose:
            return f'Error: {self.error:.2f}m'
        else:
            return self.error

    def reset(self, verbose=True):
        # store the set of targets we can select from 
        self.target_list = list(range(self.observation_space.low[0], self.observation_space.high[0] + 1))

        random.shuffle(self.target_list) # shuffle the available targets
        self.new_target() # generate a target value
        self.range = 0 # the projectile is yet to be shot, so range is still 0
        self.current_countdown = self.max_repeat # start the countdown for the current target
        self.done = False

        return self.get_obs(verbose), self.get_info()


    def calc_range(self):
        # calculate the range of the projectile using the relation v²·sin(2θ)/g
        rad = np.pi/180 * self.angle
        range = (self.v**2) * np.sin(2 * rad) / self.g
        return range

    def step(self, action):
        # print(self.target_list)
        # reduce the target repeat countdown by 1
        self.current_countdown -= 1

        # map the action to the angle of projectile
        self.angle = self.action_to_angle[action]

        # this is the range covered by the projectile at the given angle, save it in the 
        # lookup table for current ranges
        self.range = self.calc_range()

        # evaluate error for this round
        self.error = self.get_info(verbose=False) 

        # if the error in this round is less than the history of errors for this round, the 
        # reward is set to +1. If it is higher, -1. Else, the reward is 0
        if self.error < self.min_error[self.target]:
            self.reward = 1 
            self.min_error[self.target] = self.error # update error range value
        elif self.error > self.min_error[self.target]:
            self.reward = -1
        elif self.error == self.min_error[self.target]:
            self.reward = 0

        if self.current_countdown == 0:
            if len(self.target_list) == 0:
                # set self.done to True when there are no values left in self.target_list and end episode
                self.done = True
            else:
                # else supply a new target and start the repeat counter again
                self.new_target()
                self.current_countdown = self.max_repeat
        
        return self.get_obs(), self.reward, self.done, self.get_info()
        

    def render(self):
        rad = np.pi/180 * self.angle
        plt.figure()
        # plt.clf()
        params = {'mathtext.default': 'regular' }  # mathtext, for subscripting 0 in v0 in plot title
        tmax = (2 * self.v) * np.sin(rad) / self.g # calculate time of flight
        t = tmax*np.linspace(0,1,100) # divide time of flight into 100 uniform time steps
        self.x = ((self.v * t) * np.cos(rad)) # horizontal distance at each time step
        self.y = ((self.v * t) * np.sin(rad)) - ((0.5 * self.g) * (t ** 2)) # vertical distance 

        plt.plot(self.x, self.y, color='g') # plot path

        # draw line to target target (i.e desired distance) saved in `self.target`
        plt.axvline(x = self.target, ls='--', color = 'b', label = f'target: {self.target}m') 

        # draw projectile at final coordinates
        plt.scatter(self.x[-1], self.y[-1], color='r', marker="^", s=200)

        plt.ylim([0,10])
        plt.xlim(left=0)
        plt.title(f'$v_{0}$ = {self.v}m/s, θ = {self.angle}°, abs. error = {np.abs(self.x[-1]-self.target):.2f}m')
        plt.legend()
        plt.show()
    
    def __repr__(self):
        return f'''Projectile environment: 
                   Initial velocity: {self.v}m/s
                   Acceleration due to gravity: {self.g}m/s²
                   Available angles in degrees:{list(self.action_to_angle.values())}'''