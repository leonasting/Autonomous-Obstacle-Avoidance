# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt 
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
        self.ls_reward=[]
        self.reached=False
        self.path=[]
    def reset_q_val(self):
        self.reached=False
        self.q_val=np.zeros((self.side,self.side, 4))
        
    def get_new_val(self):    
        a=random.randint(0,self.side-1)
        b=random.randint(0,self.side-1)
        print(a,b)
        new=[a,b]
        return new
    
    def add_bomb(self):
        for i in range(self.bomb_count):
            new=self.get_new_val()
            while new in self.ls_filled:
                new=self.get_new_val()
            
            self.ls_filled.append(new)
            print(new)
            self.reward[new[0],new[1]]=-100
            
    def is_terminal_state(self,current_row_index, current_column_index):
  #if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
      if self.reward[current_row_index, current_column_index] == -1:
        return False
      else:
        return True  
    
    #define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
    def get_next_action(self,current_row_index, current_column_index, epsilon,exclude=False):
        #if a randomly chosen value between 0 and 1 is less than epsilon, 
        #then choose the most promising value from the Q-table for this state.
        if exclude:
            ls_action = [0,1,2,3]


            if current_row_index == 0:
                ls_action.remove(2)
            if current_column_index==0:
                ls_action.remove(3)
            if current_row_index == self.side-1:
                ls_action.remove(0)
            if current_column_index == self.side-1:
                ls_action.remove(1)
            return random.choice(ls_action)
         
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
  
    
    def q_learning(self, episode_count=1000, learning_rate=0.9, discount_factor=0.9, epsilon=0.9):
        """
        #define training parameters
    
        epsilon:the percentage of time when we should take the best action (instead of a random action)
        discount_factor:discount factor for future rewards
        learning_rate:the rate at which the AI agent should learn
        """
  
    

        #run through 1000 training episodes
        ls_episode_reward=[]
        for episode in range(episode_count):
            ls_path=[]
            #get the starting location for this episode
            row_index , column_index = self.start[0],self.start[1]
            #continue taking actions (i.e., moving) until we reach a terminal state
            #(i.e., until we reach the item packaging area or crash into an item storage location)
            reward_of_episode = 0
            while not self.is_terminal_state(row_index, column_index):#For each Step
                #choose which action to take (i.e., where to move next)
                action_index = self.get_next_action(row_index, column_index, epsilon)
        
                #perform the chosen action, and transition to the next state (i.e., move to the next location)
                old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
                ls_path.append([row_index,column_index])
                
                row_index, column_index = self.get_next_location(row_index, column_index, action_index)
                if row_index==self.goal[0] and column_index==self.goal[1]:
                    self.reached=True
                    self.path=ls_path
                
                if row_index==self.goal[0] and column_index==self.goal[1]:
                    self.reached=True

                #receive the reward for moving to the new state, and calculate the temporal difference
                reward = self.reward[row_index, column_index]
                reward_of_episode+=reward
                old_q_value = self.q_val[old_row_index, old_column_index, action_index]
                temporal_difference = reward + (discount_factor * np.max(self.q_val[row_index, column_index])) - old_q_value
        
                #update the Q-value for the previous state and action pair
                new_q_value = old_q_value + (learning_rate * temporal_difference)
                self.q_val[old_row_index, old_column_index, action_index] = new_q_value
            ls_episode_reward.append(reward_of_episode)
        print('Training complete!')
        self.ls_reward=ls_episode_reward
        if row_index==self.goal[0] and column_index==self.goal[1]:
            self.reached=True
            self.path=ls_path

    def SARSA(self, episode_count=1000, learning_rate=0.9, discount_factor=0.9, epsilon=0.99):
        # run through 1000 training episodes
        ls_episode_reward = []

        for episode in range(episode_count):
            ls_path=[]
            # get the starting location for this episode
            row_index, column_index = self.start[0], self.start[1]

            reward_of_episode = 0
            # Choose action from current position using policy derived from (e-greedy)
            action_index = self.get_next_action(row_index, column_index, epsilon)
            while not self.is_terminal_state(row_index, column_index):  # For each Step
                # Saving old position
                old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
                old_action_index = action_index
                ls_path.append([row_index,column_index])
                # perform the chosen action, and transition to the next state (i.e., move to the next location)
                row_index, column_index = self.get_next_location(row_index, column_index, action_index)
                if row_index==self.goal[0] and column_index==self.goal[1]:
                    self.reached=True
                    self.path=ls_path

                # receive the reward for moving to the new state, and calculate the temporal difference
                reward = self.reward[row_index, column_index]
                reward_of_episode += reward
                old_q_value = self.q_val[old_row_index, old_column_index, action_index]

                action_index = self.get_next_action(row_index, column_index, epsilon)
                if row_index==old_row_index and column_index == old_column_index:
                    continue
                    row_index, column_index = self.get_next_location(row_index, column_index, action_index,exclude_opt=True)

                # SARSA Formula : temporal_difference = Reward + (Discount * Q(S', A') - Q(S,A))
                temporal_difference = reward + (
                            discount_factor * self.q_val[row_index, column_index, action_index]) - old_q_value

                # update the Q-value for the previous state and action pair
                new_q_value = old_q_value + (learning_rate * temporal_difference)
                self.q_val[old_row_index, old_column_index, old_action_index] = new_q_value

            ls_episode_reward.append(reward_of_episode)
        print('Training complete!')
        self.ls_reward = ls_episode_reward
        if self.reached==False:
            self.path=ls_path
    def get_q_val_policy(self):
        ls_row=[]
        for i in range(self.side):
            ls_col=[]
            for j in range(self.side):
                ls_col.append(round(max(self.q_val[i][j]),2))
            ls_row.append(ls_col)
        
        df_q_val=pd.DataFrame(ls_row)
        actions_symbol = ['^', '>', 'v', '<']
        ls_row=[]
        for i in range(self.side):
            ls_col=[]
            for j in range(self.side):
                ls_col.append(actions_symbol[np.argmax(self.q_val[i][j])])
            ls_row.append(ls_col)
        
        for i in self.ls_filled:
            if i==self.start:
                continue
                
            ls_row[i[0]][i[1]]='B'
        
        ls_row[self.start[0]][self.start[1]]=ls_row[self.start[0]][self.start[1]]+'S'
        ls_row[self.goal[0]][self.goal[1]]='G'
            
        df_pol=pd.DataFrame(ls_row)
            
        
        return df_q_val,df_pol
    
    def display_path(self):
        actions_symbol = ['^', '>', 'v', '<']
        ls_row = []
        for i in range(self.side):
            ls_col = []
            for j in range(self.side):
                if [i, j] in self.path:
                    ls_col.append(actions_symbol[np.argmax(self.q_val[i][j])])
                else:
                    ls_col.append(' ')
            ls_row.append(ls_col)

        for i in self.ls_filled:
            if i == self.start:
                continue

            ls_row[i[0]][i[1]] = 'B'

        ls_row[self.start[0]][self.start[1]] = ls_row[self.start[0]][self.start[1]] + 'S'
        ls_row[self.goal[0]][self.goal[1]] = 'G'

        df_pol = pd.DataFrame(ls_row)

        #ls_row = [[' ' for i in range(self.side)] for j in range(self.side)]

        #for i in range(len(self.path)-1):
        #    self.path[i]




        return df_pol


if __name__=="__main__":


    # ls_filled=[[1,2]]
    # new=get_new_val()
    # while new in ls_filled:
    #     new=get_new_val()
        
    # ls_filled.append(new)
    new_grid=GridPOMDP(7,7,start=[2,2],goal=[5,5])
    new_grid.add_bomb()
    # new_grid.q_learning()
    new_grid.q_learning(episode_count=1000)
    print(new_grid.display_path())
    
    q_reward=new_grid.ls_reward
    new_grid.reset_q_val()
    
    new_grid.SARSA(episode_count=1000)
    print(new_grid.display_path())
    sarsa_reward=new_grid.ls_reward
    
    new_grid.q_val
    df_q_val,df_pol=new_grid.get_q_val_policy()
    df_reward=pd.DataFrame(new_grid.reward)
    print(df_pol)
    print(df_q_val)
    print(new_grid.ls_reward[-6:])
    print(new_grid.reached)
    print(new_grid.path)
    print(new_grid.display_path())
    df_res = pd.DataFrame({"Q- Learning Accumulated reward":q_reward,"SARSA Accumulated reward":sarsa_reward})
     
    
    plt.figure()
    df_res.plot(title='Q-learning vs SARSA Accumulated Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Accumulated Rewards')
    plt.show()
    
    