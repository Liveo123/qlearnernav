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
        # NB We can't just use all 3 items in a tuple ie (s,a,s') because later
        # we need to search for (s, a, *) i.e. whether there are any tuples where 
        # s and a are in there irrelevant of the value of s'.  3D arrays are out
        # because we would need to store loads of unused states, which would
        # make it pretty inefficient.  Need to use a dictionary of arrays of
        # tuples.  so, any states could have any number of resulting states
        # and each contains a counter of the number of times used e.g.:
        # (1, 4)[0] = 20   (1,4)[1] = 31 (1,4)[3] = 2
        # Here transtion (1,4)[2] has never occurred yet.
        self.T = {}

        # Set up some tests:
        self.T[(90,0)] = {0: 2}
        self.T[(90,0)][1] = 3
        self.T[(90,0)][3] = 5
        self.T[(90,1)] = {0: 2}
        self.T[(90,1)][1] = 3
        self.T[(80,0)] = {1: 3}

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
       
        #### DYNA-Q

        # First we need to check whether the transition table has had this combo
        # of state, action which results in s_prime.  If it hasn't, add it.
        # If it has, if it has, then increment it.
        if self.verbose: print( "self.s = %d self.a = %d" % (self.s, self.a))
        if (self.s, self.a) not in self.T:
            self.T[(self.s, self.a)] = {s_prime: 1}
        # Now check if each (s, a) tuple has a s_prime in the 
        # data structure.  If it hasn't create one...
        else:   
            if s_prime not in self.T[(self.s, self.a)]:
                self.T[(self.s, self.a)][s_prime] = 1
            else:
                #else increment the one that is there
                self.T[(self.s, self.a)][s_prime] += 1

        # Create a local copy of the Q-table
        Q_copy = np.copy(self.Q)

        # Now let's hit the Dyna loop:
        for cnt in range(self.dyna):
            # Find random numbers for the new state and action
            s_new = rand.randint(0, self.num_states - 1)
            a_new = rand.randint(0, self.num_actions- 1)
            s_new = 90
            a_new = 0
            if(s_new, a_new) in self.T:
                max_cnt = 0
                max_s_prime = -1
                # Go through all of the s_primes in the transition table 
                # for s_new, a_new and find the biggest.  This is the one
                # we will use b/c as everybody knows, biggest is best.
                for trans_cnt in self.T[(s_new, a_new)]:
                    if self.T[(s_new, a_new)][trans_cnt] > max_s_prime:
                        max_s_prime = trans_cnt                         
                        max_cnt = self.T[(s_new, a_new)][trans_cnt]

                
                #if self.verbose: print ("max_cnt = %d  max_s_prime = %d" % (max_cnt, max_s_prime))
                #if self.verbose: print "self.T ="
                #if self.verbose: print self.T
                
                # Update the Q-table:
                Q_copy[s_new, a_new] = ((1 - self.alpha)  * 
                                       Q_copy[s_new, a_new]) + \
                                       (self.alpha * (r + (self.gamma * \
                                        Q_copy[max_s_prime, 
                                            Q_copy[max_s_prime, :].argmax()])))
                # Copy the Q-table back again...
                self.Q = np.copy(Q_copy)


        # 4. Update rar = rar * radr
        self.rar = self.rar * self.radr


        # 4. Update rar = rar * radr
        self.rar = self.rar * self.radr
        self.s = s_prime
        
        self.a = action
    
        if self.verbose: 
            print "s =", s_prime,"a =",action,"r =",r
        
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
