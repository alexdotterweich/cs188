# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
import time

class AsynchronousValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        
        i = 0
        while i < self.iterations:
            start = time.time()
            index = i % len(states)
            state = states[index]
            if not mdp.isTerminal(state):
                maxOverActions = -99999
                for action in mdp.getPossibleActions(state):
                    result = 0
                    for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                        result += prob * (mdp.getReward(state) + (self.discount * self.values[nextState]))
                    #find max action
                    if result > maxOverActions:
                        maxOverActions = result     
                self.values[state] = maxOverActions
            print time.time() - start
            i += 1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        result = 0
        if self.mdp.isTerminal(state):
            return 0
        tranStateP = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in tranStateP:
            result += prob * (self.mdp.getReward(state) + (self.discount * self.values[nextState]))
        return result

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        i = 0
        bestVal = -99999
        bestAction = None
        actions = self.mdp.getPossibleActions(state)
        results = [self.computeQValueFromValues(state, action) for action in actions]
        while i < len(results):
            if results[i] > bestVal:
                bestVal = results[i]
                bestAction = actions[i]
            i += 1    
        return bestAction

    def computeHighestQValueAcrossActions(self, state):
        i = 0
        bestVal = -99999
        bestAction = None
        actions = self.mdp.getPossibleActions(state)
        results = [self.computeQValueFromValues(state, action) for action in actions]
        while i < len(results):
            if results[i] > bestVal:
                bestVal = results[i]
                bestAction = actions[i]
            i += 1    
        return bestVal

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0
        "*** YOUR CODE HERE ***"
        i = 0 
        preds = {}
        pQ = util.PriorityQueue() 

        for state in states:
            #make pred list
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                tranStateP = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, prob in tranStateP:
                    if not nextState in preds:
                        preds[nextState] = set([])
                    preds[nextState].add(state)

            #find diff and push to pq
            if not mdp.isTerminal(state):
                currStateValue = self.values[state]
                highestQValue = self.computeHighestQValueAcrossActions(state)
                diff = abs(currStateValue - highestQValue)
                pQ.push(state, -diff)

        while i < self.iterations:
            start = time.time()
            if not pQ.isEmpty():
                s = pQ.pop()
                if not self.mdp.isTerminal(s):
                    #update self.values
                    maxOverActions = -99999
                    for action in mdp.getPossibleActions(s):
                        result = 0
                        for nextState, prob in mdp.getTransitionStatesAndProbs(s, action):
                            result += prob * (mdp.getReward(s) + (self.discount * self.values[nextState]))
                        #find max action
                        if result > maxOverActions:
                            maxOverActions = result     
                    self.values[s] = maxOverActions

                for pred in preds[s]:
                    currValP = self.values[pred]
                    highestQValP = self.computeHighestQValueAcrossActions(pred)
                    diff = abs(currValP - highestQValP)
                    if diff > theta:
                        pQ.update(pred, -diff)
            print time.time() - start
            i += 1

