# Array maths 
import numpy as np 

# Ploting 
import matplotlib.pyplot as plt

# Typehinting 
from typing import Tuple 


# Class for the grid world
class GridWorld: 

    def __init__(
        self, 
        n: int, 
        goal_reward: float,
        step_reward: float,
        gamma: float,
        ): 
        """
        Constructor for the GridWorld RL problem
        
        Arguments
        ---------
        n: int
            Size of the grid world
        goal_reward: float
            Reward for reaching the goal
        step_reward: float
            Reward for taking a step
        gamma: float
            Discount factor
        """
        self.n = n 
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.gamma = gamma 

        # Initiating an empty grid world; The bellow matrix will be used as the reward matrix
        self.G = np.zeros((n, n)) 
        self.G[self.G == 0] = step_reward

        # Initiating the empty value array 
        self.V = np.zeros((n, n))

        # Creating the state array
        self.S = np.arange(0, self.n * self.n).reshape(self.n, self.n)

        # Creating the policy dictionary 
        self.create_policy()

    def create_policy(self):
        # Saving all the unique states to a vector 
        states = np.unique(self.S)

        # Dictionary to hold each action for a given state
        P = {}
        for s in states: 
            s_dict = {}

            # Checking which index is the current state in the S matrix 
            s_index = np.where(self.S == s)

            # If the state is in the top left corner, we can only move right and down
            if s_index == (0, 0):
                s_dict['right'] = 0.5
                s_dict['down'] = 0.5
            
            # If the state is in the top right corner, we can only move left and down
            elif s_index == (0, self.n - 1):
                s_dict['left'] = 0.5
                s_dict['down'] = 0.5
            
            # If the state is in the bottom left corner, we can only move right and up
            elif s_index == (self.n - 1, 0):
                s_dict['right'] = 0.5
                s_dict['up'] = 0.5
            
            # If the state is in the bottom right corner, we can only move left and up
            elif s_index == (self.n - 1, self.n - 1):
                s_dict['left'] = 0.5
                s_dict['up'] = 0.5
            
            # If the state is in the first row, we can only move left, right, and down
            elif s_index[0] == 0:
                s_dict['left'] = 0.333
                s_dict['right'] = 0.333
                s_dict['down'] = 0.333
            
            # If the state is in the last row, we can only move left, right, and up
            elif s_index[0] == self.n - 1:
                s_dict['left'] =  0.333
                s_dict['right'] = 0.333
                s_dict['up'] = 0.333
            
            # If the state is in the first column, we can only move up, down, and right
            elif s_index[1] == 0:
                s_dict['up'] = 0.333
                s_dict['down'] = 0.333
                s_dict['right'] = 0.333
            
            # If the state is in the last column, we can only move up, down, and left
            elif s_index[1] == self.n - 1:
                s_dict['up'] = 0.333
                s_dict['down'] = 0.333
                s_dict['left'] = 0.333

            # If the state is in the middle, we can move in all directions
            else:
                s_dict['up'] = 0.25
                s_dict['down'] = 0.25
                s_dict['left'] = 0.25
                s_dict['right'] = 0.25

            # Saving the current states trasition probabilities
            P[s] = s_dict
        
        # Saving the policy to the class
        self.P = P

    def add_random_goal(self): 
        """
        Function that adds the goal value to a random position in the grid
        """
        # Randomly choosing a position
        x = np.random.randint(self.n)
        y = np.random.randint(self.n)

        # Adding the goal value
        self.G[x, y] = self.goal_reward

    def add_goal(self, x: int, y: int):
        """
        Function that adds the goal value to a given position in the grid

        Arguments
        ---------
        x: int
            x position of the goal
        y: int
            y position of the goal
        """
        self.G[x, y] = self.goal_reward

    def get_next_state(self, a: str, s: int): 
        """ 
        Function that returns the next state's coordinates given an action and a state 
        """
        # Getting the current indexes 
        s_index = np.where(self.S == s)
        s_row = s_index[0][0]
        s_col = s_index[1][0]

        # Defining the indexes of the next state
        next_row = s_row 
        next_col = s_col

        if a == 'up':
            next_row = s_row - 1
            next_col = s_col
        elif a == 'down':
            next_row = s_row + 1
            next_col = s_col
        elif a == 'left':
            next_row = s_row
            next_col = s_col - 1
        elif a == 'right':
            next_row = s_row
            next_col = s_col + 1

        return next_row, next_col

    def bellman_value(
        self,
        s: int
        ) -> Tuple: 
        """
        Calculates the conditional probability of a given state and action of returning a reward and the given next state 
        """
        # Extracting all the available actions for the given state
        actions = self.P[s]

        # Placeholder to hold the sum 
        sum = 0
        for action in actions: 
            # Extracting the probability of the given action 
            prob = actions[action]

            # Getting the next states indexes
            next_row, next_col = self.get_next_state(action, s)

            # Extracting the expected reward 
            reward = self.G[next_row, next_col]

            # Extracting the value of the next state
            value_prime = self.V[next_row, next_col]

            # Adding to the sum 
            sum += prob * (reward + self.gamma * value_prime)

        return sum

    def find_best_action(
        self,
        s: int
        ) -> Tuple:
        """
        Finds the best action given a state

        Returns the best action the Bellman value  
        """ 
        # Extracting all the possible actions for the given state
        actions = self.P[s]

        # Initial maximum value 
        current_max = -np.inf

        # The best action is the first action in the dictionary 
        best_action = list(actions.keys())[0]

        # Iterating over the actions 
        for action in actions: 
            # Getting the Bellman's value 
            value = self.bellman_value(s) 

            if value > current_max: 
                current_max = value
                best_action = action

        return best_action, current_max

    def value_iteration(
        self,
        epsilon: float = 0.0001,
        value_rounding: int = 2,
        verbose: bool = False
        ) -> None: 
        """
        Function that performs the value iteration algorithm

        The function updates the V matrix inplace 
        """
        # Iteration tracker 
        iteration = 0

        # Iterating until the difference between the value functions is less than epsilon 
        iterate = True
        while iterate: 
            # Placeholder for the maximum difference between the value functions 
            delta = 0
            
            # Updating the iteration tracker
            iteration += 1 
            # Iterating over the states 
            for s in self.S.flatten():
                
                # Getting the indexes of s in S 
                s_index = np.where(self.S == s)
                s_row = s_index[0][0]
                s_col = s_index[1][0]

                # Saving the current value for the state
                v_init = self.V[s_row, s_col].copy()

                # Getting the best action and the Bellman's value 
                _, bellman_value = self.find_best_action(s)

                # Updating the value function with a rounded value
                self.V[s_row, s_col] = np.round(bellman_value, value_rounding)

                # Updating the delta 
                delta = np.max([delta, np.abs(self.V[s_row, s_col] - v_init)])

                if verbose: 
                    print(f"Iteration {iteration} - State {s} - Delta {delta}")

                if delta < epsilon: 
                    iterate = False
                    break

        return None

    def update_policy(self): 
        """
        Function that updates the policy given the value function 
        """
        # Iterating over the states 
        for s in self.S.flatten(): 
            # Listing all the actions 
            actions = self.P[s]

            # For each available action, getting the Bellman's value
            values = {}
            for action in actions.keys():
                # Getting the next state indexes
                next_row, next_col = self.get_next_state(action, s)

                # Saving the value function of that nex t state
                values[action] = self.V[next_row, next_col]
            
            # Extracting the maximum key value of the values dictionary 
            max_value = max(values.values())        

            # Leaving the keys that are equal to the maximum value
            best_actions = [key for key in values if values[key] == max_value]

            # Getting the length of the dictionary 
            length = len(values)

            # Creating the final dictionary with all the best actions in it 
            p_star = {}
            for action in best_actions:
                p_star[action] = 1/length

            # Updating the policy 
            self.P[s] = p_star

    @staticmethod
    def plot_matrix(
        M: np.array, 
        img_width: int = 5, 
        img_height: int = 5, 
        title: str = None,
        file_output: str = None
        ) -> None: 
        """
        Plots a matrix as an image.
        """
        height, width = M.shape

        fig = plt.figure(figsize=(img_width, img_height))
        ax = fig.add_subplot(111, aspect='equal')
        
        for y in range(height):
            for x in range(width):
                ax.annotate(str(M[y][x]), xy=(x, height - y - 1), ha='center', va='center')

        offset = .5    
        ax.set_xlim(-offset, width - offset)
        ax.set_ylim(-offset, height - offset)

        ax.hlines(y=np.arange(height+1)- offset, xmin=-offset, xmax=width-offset)
        ax.vlines(x=np.arange(width+1) - offset, ymin=-offset, ymax=height-offset)

        plt.title(title)
        if file_output is not None: 
            plt.savefig(file_output)
            plt.close()
        else:
            plt.show()

    def plot_states(self, **kwargs) -> None:
        """
        Plots the states matrix
        """
        self.plot_matrix(self.S, **kwargs)
    
    def plot_rewards(self, **kwargs) -> None:
        """
        Plots the rewards matrix
        """
        self.plot_matrix(self.G, **kwargs)
    
    def plot_values(self, **kwargs) -> None:
        """
        Plots the values matrix
        """
        self.plot_matrix(self.V, **kwargs)

    def plot_policy(
        self, 
        img_width: int = 5, 
        img_height: int = 5, 
        title: str = None, 
        file_output: str = None
        ) -> None: 
        """ 
        Plots the policy matrix out of the dictionary provided; The dictionary values are used to draw the arrows 
        """
        height, width = self.S.shape

        fig = plt.figure(figsize=(img_width, img_height))
        ax = fig.add_subplot(111, aspect='equal')
        for i in range(height):
            for j in range(width):
                # Adding the arrows to the plot
                if 'up' in self.P[self.S[i, j]]:
                    plt.arrow(j, i, 0, -0.3, head_width = 0.05, head_length = 0.05)
                if 'down' in self.P[self.S[i, j]]:
                    plt.arrow(j, i, 0, 0.3, head_width = 0.05, head_length = 0.05)
                if 'left' in self.P[self.S[i, j]]:
                    plt.arrow(j, i, -0.3, 0, head_width = 0.05, head_length = 0.05)
                if 'right' in self.P[self.S[i, j]]:
                    plt.arrow(j, i, 0.3, 0, head_width = 0.05, head_length = 0.05)

        offset = .5    
        ax.set_xlim(-offset, width - offset)
        ax.set_ylim(-offset, height - offset)

        ax.hlines(y=np.arange(height+1)- offset, xmin=-offset, xmax=width-offset)
        ax.vlines(x=np.arange(width+1) - offset, ymin=-offset, ymax=height-offset)

        plt.title(title)
    
        if file_output is not None: 
            plt.savefig(file_output)
            plt.close()
        else:
            plt.show()
