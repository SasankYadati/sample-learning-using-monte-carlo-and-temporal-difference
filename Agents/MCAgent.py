from Agents.Agent import Agent
import numpy as np
import random
from utils import argmax

class MCAgent(Agent):
    """
    Monte Carlo agent learns from samples without any explicit model.
    Computes value functions by averaging sample returns.
    """
    def __init__(self, num_states, num_actions, get_state_rep=None, discount_rate=1.0, epsilon=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.value_fn = np.zeros((self.num_states, self.num_actions))
        self.policy = []
        self.get_state_rep = get_state_rep
        for _ in range(self.num_states):
            p_s = [0] * self.num_actions
            random_action = random.randint(0, self.num_actions-1)
            p_s[random_action] = 1
            self.policy.append(p_s)

    def policy_evaluation(self, episodes):
        """
        Compute action value function using first-visit MC prediction for a given list of episodes.
        """
        counts = np.zeros((self.num_states, self.num_actions))
        for episode in episodes:
            returns_eps, counts_eps = self.policy_evaluation_for_episode(episode)
            for state in range(self.num_states):
                for action in range(self.num_actions):
                    self.value_fn[state][action] = self.value_fn[state][action] * counts[state][action] + returns_eps[state][action]
                    self.value_fn[state][action] /= (counts[state][action] + 1)
            counts = np.add(counts, counts_eps)

            

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
        debug and print(self.value_fn[state])
        if (random.random() <= self.epsilon):
            action = argmax(self.value_fn[state])
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