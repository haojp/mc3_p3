"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        #my code below
        self.exp_list = []
        self.num_states = num_states
        #self.Q = np.zeros((num_states, num_actions))
        self.Q = np.random.uniform(-1.0, 1.0, [num_states,num_actions])
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.T = np.full((num_states, num_actions, num_states), 0.00001)
        self.R = np.copy(self.Q)
        self.Tc = np.copy(self.T)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        self.a = action = np.argmax(self.Q[s])
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
v        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new reward
        @returns: The selected action
        """

        #my code below
        
        #update Q
        self.update_Q(self.s, self.a, s_prime, r)
        '''
        #incrment Tc
        self.Tc[self.s, self.a, s_prime] += 1
        self.update_model(s_prime, r)
        '''
        #update experience list
        self.exp_list.append((self.s, self.a, s_prime, r))
        #dyna
        self.run_dyna()
        #prepare for next query
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)
            print 'RANDOM ACTION!'
        else:
            action = np.argmax(self.Q[s_prime])
        self.rar *= self.radr
        if self.verbose: print "s =", s_prime,"a =", action,"r =",r
        self.s = s_prime
        self.a = action
        return action

    def update_Q(self, s, a, s_prime, r):
        action = np.argmax(self.Q[s_prime])
        #future rewards = Q[s_prime, argmax_a_prime(Q[s_prime, a_prime])]
        fut_r = self.Q[s_prime, action]
        self.Q[s, a] = (1-self.alpha)*self.Q[s,a]+\
                                 self.alpha*(r+self.gamma*fut_r)

    def update_model(self, s_prime, r):
        #T[s,a,s_prime] = Tc[s,a,s_prime]/np.sum(Tc[s,a,:])
        s_prime_sum = np.sum(self.Tc[self.s, self.a, :])
        #need to loop over all the elements and update the probabilities
        #for s_prime_temp in range(0, self.Tc.shape[2]):
        #    self.T[self.s, self.a, s_prime_temp] = self.Tc[self.s, self.a, s_prime_temp] / s_prime_sum
        #vectorize
        self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :] / s_prime_sum

        #R[s,a] = (1-alpha) * R[s,a] + alpha * r
        self.R[self.s, self.a] = (1-self.alpha) * self.R[self.s, self.a] + self.alpha * r

    def run_dyna(self):
        exp_list_len = len(self.exp_list)
        random_list = np.random.randint(exp_list_len, size=self.dyna)
        for i in range(0, self.dyna):
            temp_tuple = self.exp_list[random_list[i]]
            s = temp_tuple[0]
            a = temp_tuple[1]
            s_prime = temp_tuple[2]
            r = temp_tuple[3]
            self.update_Q(s, a, s_prime, r)
            
            

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
