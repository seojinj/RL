import numpy as np
from msdm.domains import GridWorld
from msdm.core.problemclasses.mdp import TabularPolicy,TabularMarkovDecisionProcess
from msdm.core.assignment import AssignmentMap
import matplotlib.pyplot as plt
from scipy.special import softmax, logsumexp
from msdm.core.algorithmclasses import Plans, Result
from msdm.domains.gridworld.plotting import GridWorldPlotter
from matplotlib.patches import Rectangle, Arrow, Circle

class ValueIteration:
    def __init__(self, env, th = 0.01, d = 0.99):
        # pull input parameters
        self.th = th
        self.d = d
        self.max_iter = 200
        self.entreg = True
        
        # set up environment
        self.mdp = env
        self.S = self.mdp.state_list # set of all states from MDP
        self.A = self.mdp.action_list # set of all actions from MDP
        self.P = self.mdp.transitionmatrix# state transition function
        # P is sparse: vector representation > huge (think of as function)
        self.R = self.mdp.rewardmatrix# reward function
        
    def planOn(self):
        return self.valueIteration()
        
    def valueIteration(self):
        # initialize
        S = self.S
        A = self.A
        P = self.P
        R = self.R
        V = AssignmentMap([(s,0) for s in S])
        
        for count in range(self.max_iter):
            dif = 0  
            # update value for each state
            for s in S:
                #print(s)
                v = V[s]
                # go through all actions
                qi = np.zeros(len(A))
                Q = AssignmentMap([(a,0) for a in A])
                # print(Q)
                for a in A:
                    # p = probabilty, n = next state, r = reward
                    s_i = S.index(s)
                    a_i = A.index(a)
                    for i in range(len(S)):
                        #print(a.keys())
                        old_q = Q[a]
                        p = P[s_i][a_i][i]
                        n = S[i]
                        r = R[s_i][a_i][i]
                        #if r != 0:
                            #print(p,n,r)
                        new_q = old_q + p * (r + d * V[n])
                        Q[a] = new_q
                        #print(new_q)
                max_q = -float("inf")
                #print(Q)
                for k,q in Q.items():
                    if q > max_q:
                        max_q = q
                V[s] = max_q
                dif = max(dif, abs(v - V[s]))
                #print(dif)
            #print(v,V[s])
            if dif < th:
                break

        # create policy
        policy = AssignmentMap()
        Q = AssignmentMap()
        for s in S:
            Q[s] = AssignmentMap([(a,0) for a in A])
            for a in A:
                # p = probabilty, n = next state, r = reward
                s_i = S.index(s)
                a_i = A.index(a)
                for i in range(len(S)):
                        #print(a.keys())
                        old_q = Q[s][a]
                        p = P[s_i][a_i][i]
                        n = S[i]
                        r = R[s_i][a_i][i]
                        new_q = old_q + p * (r + d * V[n])
                        Q[s][a] = new_q
            max_q = 0
            max_idx = -1
            for k,v in AssignmentMap.items(Q[s]):
                    if v > max_q:
                        max_q = v
                        max_idx = k
            opt_action = max_idx
            policy[s] = AssignmentMap()
            policy[s][opt_action] = 1
        
        # Create results   
        """if self.entreg:
            pi = softmax((1 / 1) * q, axis=-1)
        else:
            pi = np.log(np.zeros_like(q))
            pi[q == np.max(q, axis=-1, keepdims=True)] = 1
            pi = softmax(pi, axis=-1)
        res = Result()
        res.mdp = self.mdp
        res.policy = res.pi = TabularPolicy(mdp._states, mdp._actions, policydict=policy)
        res._valuevec = V
        vf = AssignmentMap([(s, vi) for s, vi in zip(self.S, V)])
        res.valuefunc = res.V = vf
        res.actionvaluefunc = res.Q = Q
        return res"""
        
        return {'value':V, 'policy':Q}