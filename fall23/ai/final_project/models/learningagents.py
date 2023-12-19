from collections import defaultdict, Counter
import numpy as np

class QTDLearningAgent:

    def __init__(self, hyperparameters):
        '''
        Constructs QTDLearning agent with initialized hyperparameter values. QTD updates policy with
        Temporal Distance approximation of future rewards.
        '''
        for k,v in hyperparameters.items():
            setattr(self, k, v)
        self.q = defaultdict(lambda : np.zeros(self.env.action_space.n))
        self.training_error = []

    def get_action(self, state):
        #epsilon greedy policy. randomly select arbitrary action, otherwise follow policy
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        max_action = int(np.argmax(self.q[state])) #state is a np.array where each index represents an action.
        return max_action

    def update(self, state, action, nextstate, reward, terminated):
        #future q is 0 if state is terminated.
        future_q_value = (not terminated) * np.max(self.q[state])
        #td estimate of future reward value
        td = reward + self.gamma * future_q_value - self.q[state][action]
        #update with learning rate adjustment
        self.q[state][action] = self.q[state][action] + self.learningrate * td
        self.training_error.append(td)

    def decay_epsilon(self):
        #decay epsilon to .1 over 50% of episodes
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

class DoubleQLearningAgent:
    def __init__(self, hyperparameters):
        '''
        DoubleQLearningAgent implemented according to van Hasselt (2015). The idea is that there is
        two Qlearning policies that are alternated between at random. Each policy is conditioned on the
        future reward of the other. This creates a much more stable learning experience, and doubles the
        parameter space so that more representation can be learned. Instead of learning rate, we use
        1/ n where n is the number of times the state, action sequence has been observed so far.
        :param hyperparameters:
        '''
        for k,v in hyperparameters.items():
            setattr(self, k, v)
        self.q1 = defaultdict(lambda : np.zeros(self.env.action_space.n))
        self.q2 = defaultdict(lambda : np.zeros(self.env.action_space.n))
        self.n1 = Counter()
        self.n2 = Counter()
        self.training_error = []

    def get_action(self, state):
        #epsilon greedy
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        #best action is chosen as the average of both policies
        max_action = int(np.argmax((self.q1[state]+self.q2[state])/2))
        return max_action

    def update(self, state, action, nextstate, reward, terminated):
        #randomly update between choosing q1 or q2
        if np.random.random() < .5:
            self.n1[(state,action)] += 1
            max_next_action = np.argmax(self.q1[nextstate])
            future_reward = (not terminated) * self.q2[nextstate][max_next_action] #condition on q* of q2
            td = reward + self.gamma *future_reward-self.q1[state][action]
            update = self.q1[state][action] + 1/self.n1[(state,action)] * td
            self.q1[state][action] = update
        else:
            self.n2[(state,action)] += 1
            max_next_action = np.argmax(self.q2[nextstate])
            future_reward = (not terminated) * self.q1[nextstate][max_next_action] #condition on q* of q1
            td = reward + self.gamma *future_reward-self.q2[state][action]
            update = self.q2[state][action] + 1/self.n2[(state,action)] * td
            self.q2[state][action] = update
        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)