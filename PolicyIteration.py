import numpy as np
from msdm.domains import GridWorld
from msdm.core.problemclasses.mdp import TabularPolicy,TabularMarkovDecisionProcess
from msdm.core.assignment import AssignmentMap

class PolicyIteration:
    def __init__(self, env, th = 0.1, d = 0.99):
        # pull input parameters
        self.th = th
        self.d = d
        
        # set up environment
        mdp = env
        self.S = mdp.state_list # set of all states from MDP
        self.A = mdp.action_list # set of all actions from MDP
        self.P = mdp.transitionmatrix# state transition function
        # P is sparse: vector representation > huge (think of as function)
        self.R = mdp.rewardmatrix# reward function
        
        # initialize value function and policy
        vi = np.zeros(len(self.S))
        self.V = AssignmentMap([(s,v) for s,v in zip(self.S,vi)])
        self.policy = AssignmentMap([(s,self.A[0]) for s in self.S])
        
    def Iterate(self):
        while True:
            self.policyEvaluation()
            output = self.policyImprovement()
            if output["policy"] != -1:
                return output
        
        
    def policyEvaluation(self):
        # Policy evaluation
        # outputs from MDP are not guaranteed to be hashable
        # use dictionaries
        dif = 100
        S = self.S
        A = self.A
        P = self.P
        R = self.R
        V = self.V
        while dif >= self.th:
            dif = 0
            for s in S:
                v = V[s]
                new_v = 0
                s_i = S.index(s)
                a_i = A.index(self.policy[s])
                for i in range(len(S)):
                    #print(a.keys())
                    #print(self.policy[s])
                    p = P[s_i][a_i][i] # probability
                    n = S[i] # next_state
                    r = R[s_i][a_i][i] # reward
                    new_v += p * (r + self.d * V[n])
                    #print(p,n,r,policy[s],V[n])
                V[s] = new_v
                #print(v, V[s])
                dif = max(dif, abs(v - V[s])) 
                
    def policyImprovement(self):
        # Policy improvement
        policy_stable = True

        S = self.S
        A = self.A
        P = self.P
        R = self.R
        V = self.V
        
        Q = AssignmentMap()
        for s in S:
            old_action = self.policy[s]
            Q[s] = AssignmentMap([(a,0) for a in A])
            for a in A:
                # p = probabilty, n = next state, r = reward
                #init actions
                s_i = S.index(s)
                a_i = A.index(a)
                for i in range(len(S)):
                        #print(a.keys())
                        old_q = Q[s][a]
                        p = P[s_i][a_i][i] # probability
                        n = S[i] # next_state
                        r = R[s_i][a_i][i] # reward
                        Q[s][a] += p * (r + self.d * V[n])
            max_q = -float("inf")
            max_idx = -1

            for k,v in AssignmentMap.items(Q[s]):
                if v > max_q:
                    max_q = v
                    max_idx = k

            opt_action = max_idx
            self.policy[s] = opt_action

            if old_action != opt_action:
                policy_stable = False
            
            #policy[s] = np.eye(len(A))[opt_action]

        if policy_stable:
            return {"policy": self.policy, "value":V}
        else:
            return {"policy": -1, "value":-1}