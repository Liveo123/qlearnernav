"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import os

class QLearner(object):
    
    def author(self):
        return 'plivesey3'
   
    def __init__(self, \
        num_states = 100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
    
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar 
        self.radr = radr 
        self.dyna = dyna
        self.s = 0
        self.a = 0

        # Create a Q-table for the states and actions.  This contains a number
        # that represents the value of taking a in state s where 
        # Q[s,a] = value.  This is immediate reward + discounted reward (for 
        # future actions.
        self.Q = np.zeros((self.num_states, self.num_actions))

        # Create the transition table, which translates s to s'.  Dyna
        self.T = {}

        # Create the rewards table.  This gets updated on every loop of the Dyna.
        self.R = np.zeros((self.num_states, self.num_actions))


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        # We create a next action from a previous state and action.
        # If we have no previous state and action (i.e. first step or
        # when using a learned policy), then we need to create them. 
        # So this says "this is the state you are in, don't update 
        # anything and then start query"
        action = rand.randint(0, self.num_actions-1)
        self.a = action
        self.s = s
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The next state
        @param r: The immediate reward
        @returns: The selected action
        """
        # Steps that need to be taken according to the video:
        # 1. Update the Q table
        #if self.verbose: print "self.s =", self.s,"self.a =",self.a
        
        self.Q[self.s, self.a] = ((1 - self.alpha)  * self.Q[self.s, self.a]) + \
                                 (self.alpha * (r + (self.gamma * \
                                 self.Q[s_prime, self.Q[s_prime, :].argmax()])))
        # 2. Roll the dice to decide whether you should take
        # a random action or not.  If so, choose random action and
        # return.
        if rand.uniform(0, 1) < self.rar:
            if self.verbose:
                print("Using random action")              
            action = rand.randint(0, self.num_actions-1)
        # 3. Det. what action to take (if not rnd).  Go to the row 
        # s_prime (the new state you are in) and look at the Q values
        # for each of the actions you might take and choose the one 
        # with the highest value.  This is the action you will
        # take.
        else:
            action = self.Q[s_prime, :].argmax() #rand.randint(0, self.num_actions-1)
            if self.verbose:
                print("Using argmax action %d", action)              
        # 4. Update rar = rar * radr
        self.rar = self.rar * self.radr
        # 2. Remember the state you were in before and previous 
        # action
        # 3. You also have s' and r and these are all you need
        # to update your q table 
        
        self.s = s_prime
        self.a = action
    
        if self.verbose: 
            print "s =", s_prime,"a =",action,"r =",r
        
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
