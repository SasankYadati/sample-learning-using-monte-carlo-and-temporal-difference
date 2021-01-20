import numpy as np
import random
from utils import argmax

class MCAgent():
    """
    Monte Carlo agent learns from samples without any explicit model.
    Computes value functions by averaging sample returns.
    The policy is an implicit epsilon greedy policy on the current value function.
    """
    def __init__(self, num_states, num_actions, get_state_rep=None, epsilon=0.1, discount_rate=1.0, seed=42):
        self.num_states = num_states
        self.num_actions = num_actions
        self.get_state_rep = get_state_rep
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.seed = seed
        self.set_seed()
        self.value_fn = np.zeros((self.num_states, self.num_actions))
        self.counts = np.zeros((self.num_states, self.num_actions))

    def set_seed(self, seed=None):
        if seed:
            self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def policy_evaluation(self, episodes):
        """
        Compute action value function using first-visit MC prediction for a given list of episodes.
        """
        for episode in episodes:
            returns_eps, counts_eps = self.policy_evaluation_for_episode(episode)
            for state in range(self.num_states):
                for action in range(self.num_actions):
                    self.value_fn[state][action] = self.value_fn[state][action] * self.counts[state][action] + returns_eps[state][action]
                    self.value_fn[state][action] /= (self.counts[state][action] + 1)
            self.counts = np.add(self.counts, counts_eps)       

    def policy_evaluation_for_episode(self, episode):
        """
        Compute action value function using first-visit MC prediction for a given episode.
        """
        return_ = 0
        episode = episode[::-1]
        returns = np.zeros((self.num_states, self.num_actions))
        counts = np.zeros((self.num_states, self.num_actions))
        for i in range(0, len(episode)):
            s, a, r = episode[i]
            return_ = r + self.discount_rate * return_
            find_sa_pair = len(list(filter(lambda step: step[0] == s and step[1] == a, episode[i+1:]))) > 0
            if not find_sa_pair:
                returns[s][a] = return_
                counts[s][a] += 1
        return returns, counts

    def step(self, state, debug=False):
        """
        Returns the next action epsilon greedily using value function.
        """
        if self.get_state_rep:
            state = self.get_state_rep(state)
        debug and print("State", state)
        debug and print("Value function", self.value_fn[state])
        if (random.random() >= self.epsilon):
            action = argmax(self.value_fn[state], seed=self.seed)
        else:
            action = random.choice(range(self.num_actions))
        debug and print("Action", action)
        return int(action)

    def generate_episode(self, env, observation):
        """
        """
        done = False
        episode = []
        while not done:
            if self.get_state_rep:
                state = self.get_state_rep(observation)
            else:
                state = observation
            action = self.step(state)
            observation, reward, done, info  = env.step(action)
            episode.append((state, action, reward))
        return episode