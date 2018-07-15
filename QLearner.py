"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import os
import time as tm
from copy import deepcopy

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

        self.loop_time = 0
        self.total_time = 0

        # Create a Q-table for the states and actions.  This contains a number
        # that represents the value of taking a in state s where 
        # Q[s,a] = value.  This is immediate reward + discounted reward (for 
        # future actions.
        self.Q = np.zeros((self.num_states, self.num_actions))

        # Orginally had T as a dictionary, but was too slow, so
        # changed it to a numpy array.  Much, much faster now.
        self.T = np.zeros((num_states, num_actions, num_states))

        # Create counting transition matrix for Dyna-Q and fill 
        # with 0.0001 (a small number to stop division by 0)
        self.Tc = np.full((num_states, num_actions, num_states), 0.0001)

        # Create the rewards table.  This gets updated on every loop of the Dyna.
        self.R = np.zeros((self.num_states, self.num_actions))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        # Roll the dice to decide whether you should take
        # a random action or not.  If so, choose random action and
        # return.
        if self.verbose: print "self.rar =", self.rar
        if rand.uniform(0, 1) < self.rar:
            if self.verbose:
                print("Using random action")              
            action = rand.randint(0, self.num_actions-1)
        # Det. what action to take (if not rnd).  Go to the row 
        # s_prime (the new state you are in) and look at the Q values
        # for each of the actions you might take and choose the one 
        # with the highest value.  This is the action you will
        # take.
        else:
            action = self.Q[s, :].argmax() #rand.randint(0, self.num_actions-1)
            if self.verbose:
                print("Using argmax action %d", action)              

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

        # Replace all of the oop variables with local variables to speed things
        # up.  The equation to update the Q-table is very slow.  This for 
        # optimization purposes.
        c_s = self.s
        c_a = self.a
        c_alpha = self.alpha
        c_gamma = self.gamma
        c_num_states = self.num_states
        c_num_actions = self.num_actions
        c_T = self.T
        c_Tc = self.Tc
        c_R = self.R
        c_Q = self.Q
        c_dyna = self.dyna
        c_rar = self.rar
        c_radr = self.radr
        
        # Time function:
        if self.verbose: 
            time_in_func = tm.time()
            time1 = tm.time()

        # Steps that need to be taken according to the video:
        # 1. Update the Q table
        if self.verbose: print "c_s =", c_s,"c_a =",c_a
        
        c_Q[c_s, c_a] = ((1 - c_alpha)  * c_Q[c_s, c_a]) + \
                         (c_alpha * (r + (c_gamma * \
                         c_Q[s_prime, c_Q[s_prime, :].argmax()])))

        
        if self.verbose: print "self.Q =", c_Q
        if self.verbose: print "self.Q[self.s, self.a] =", c_Q[c_s, c_a]
        
        # First we need to check whether the transition table has had this combo
        # of state, action which results in s_prime.  If it hasn't, add it.
        # If it has, if it has, then increment it.
        #if self.verbose: print( "self.s = %d self.a = %d" % (self.s, self.a))
        #if self.verbose: print "self.T before =", self.T
        
        #if (c_s, c_a) not in c_T:
            #c_T[(c_s, c_a)] = {s_prime: 1}
        # Now check if each (s, a) tuple has a s_prime in the 
        # data structure.  If it hasn't create one...
        #else:   
            #if s_prime not in c_T[(c_s, c_a)]:
                #c_T[(c_s, c_a)][s_prime] = 1
            #else:
                # else increment the one that is there
                #_T[(c_s, c_a)][s_prime] += 1

        #if self.verbose: print "self.T after =", self.T
        
        
        # Create a local copy of the Q-table
        #Q_copy = np.copy(c_Q)
        
        if self.verbose: print "inside time1 = %0.7f", tm.time() - time1
        time1 = tm.time()
        # Now let's hit the Dyna loop:
        #for cnt in range(c_dyna):
            # Find random numbers for the new state and action
            #s_new = rand.randint(0, c_num_states - 1)
            #a_new = rand.randint(0, c_num_actions- 1)
            #if(s_new, a_new) in c_T:
                #max_cnt = 0
                #max_s_prime = -1
                # Go through all of the s_primes in the transition table 
                # for s_new, a_new and find the biggest.  This is the one
                # we will use b/c as everybody knows, biggest is best.
                #for trans_cnt in c_T[(s_new, a_new)]:
                    #if c_T[(s_new, a_new)][trans_cnt] > max_s_prime:
                        #max_s_prime = trans_cnt                         
                        #max_cnt = c_T[(s_new, a_new)][trans_cnt]
                 #s_prime = max(self.T[(s_new, a_new)], key = lambda k: self.T[(s_new, a_new)][k])

                
                #if self.verbose: print ("max_cnt = %d  max_s_prime = %d" % (max_cnt, max_s_prime))
                #if self.verbose: print "self.T ="
                #if self.verbose: print c_T
                #
        
        # DYNA-Q
        if c_dyna > 0:
            # Incremement the count in the Transition array count
            c_Tc[c_s, c_a, s_prime] += 1

            # Update the Transition matrix...
            Tc_sum = np.sum(c_Tc[c_s, c_a,:])
            c_T[c_s, c_a, :] = c_Tc[c_s, c_a, :] / Tc_sum

            # ... and the rewards
            c_R[c_s, c_a] = (1 - c_alpha) * c_R[c_s, c_a] + c_alpha * r

            # Now repeat for each hallucination required
            for cnt in range(c_dyna):

                # Grab a random state
                s_new = rand.randint(0, c_num_states - 1)
                a_new = rand.randint(0, c_num_actions- 1)
        
                # Get the reward and new state from the tables
                r_new = c_R[s_new, a_new]
                s_prime_new = c_T[s_new, a_new, :].argmax()
                
                # Update the Q-tables
                c_Q[s_new, a_new] = ((1 - c_alpha)  * 
                                      c_Q[s_new, a_new]) + \
                                      (c_alpha * (r_new + (c_gamma * \
                                      c_Q[s_prime_new, 
                                          c_Q[s_prime_new, :].argmax()])))

        if self.verbose: 
            print "inside time2 = %0.7f", tm.time() - time1
            time1 = tm.time()
        
        # Finish timing and output function time and total time
        if self.verbose:
            time_in_func = tm.time() - time_in_func
            self.loop_time += time_in_func

        if self.verbose: 
            print "time_in_func = %.7f" % time_in_func
            print "self.loop_time = %.2f" % self.loop_time

        # Copy all of the local variable back to the object
        self.s = c_s
        self.a = c_a 
        self.alpha = c_alpha 
        self.gamma = c_gamma 
        self.num_states = c_num_states 
        self.num_actions = c_num_actions
        self.T = c_T 
        self.R = c_R 
        self.Q = c_Q 
        self.dyna = c_dyna 
        self.rar = c_rar 
        self.radr = c_radr 
        
        # Get the next action (either random or from table)
        action = self.querysetstate(s_prime)

        # 4. Update rar = rar * radr
        self.rar = self.rar * self.radr
        
        if self.verbose: 
            print "inside time3 = %0.7f", tm.time() - time1
        
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
