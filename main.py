# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import random

class GridPOMDP(object):
    def __init__(self,side,bomb_count,start=[-1,-1],goal=[-1,-1]):
    #N= count of states-225[]
        self.side=side
        self.bomb_count=bomb_count
        self.start=start
        self.goal=goal
        self.reward=np.full((self.side,self.side),-1.)
        
        self.reward[goal[0],goal[0]]=10000
        self.ls_filled=[start,goal]
        self.q_val=np.zeros((self.side,self.side, 4))
        #numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
        self.actions = ['up', 'right', 'down', 'left']
        
    def get_new_val(self):    
        a=random.randint(0,self.side-1)
        b=random.randint(0,self.side-1)
        new=[a,b]
        return new
    
    def add_bomb(self):
        for i in range(self.bomb_count):
            new=get_new_val()
            while new in self.ls_filled:
                new=get_new_val()
            
            self.ls_filled.append(new)
            self.reward[new[0],new[1]]=-100
            
    def is_terminal_state(self,current_row_index, current_column_index):
  #if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
      if self.reward[current_row_index, current_column_index] == -1:
        return False
      else:
        return True  
    
    #define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
    def get_next_action(self,current_row_index, current_column_index, epsilon):
  #if a randomly chosen value between 0 and 1 is less than epsilon, 
  #then choose the most promising value from the Q-table for this state.
      if np.random.random() < epsilon:
        return np.argmax(self.q_val[current_row_index, current_column_index])
      else: #choose a random action
        return np.random.randint(4)
    
    #define a function that will get the next location based on the chosen action
    def get_next_location(self,current_row_index, current_column_index, action_index):
      new_row_index = current_row_index
      new_column_index = current_column_index
      if self.actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
      elif self.actions[action_index] == 'right' and current_column_index < self.side - 1:
        new_column_index += 1
      elif self.actions[action_index] == 'down' and current_row_index < self.side - 1:
        new_row_index += 1
      elif self.actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
        
      return new_row_index, new_column_index
    


if __name__=="__main__":


    # ls_filled=[[1,2]]
    # new=get_new_val()
    # while new in ls_filled:
    #     new=get_new_val()
        
    # ls_filled.append(new)
    new_grid=GridPOMDP(3,1,start=[0,0],goal=[2,2]) 
    new_grid.add_bomb()
    
    
    
