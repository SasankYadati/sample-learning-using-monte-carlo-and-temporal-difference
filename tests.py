from Agents.MCAgent import MCAgent
from utils import argmax, get_state_rep_func
import unittest
import numpy as np

class TestPolicyEvaluationForEpisode(unittest.TestCase):
    def setUp(self):
        num_states = 2
        num_actions = 2
        self.agent = MCAgent(2, 2, 1)

    def test_short_episode(self):
        episode = [(0, 1, -10), (1, 0, 30)]
        actual_values, actual_counts = self.agent.policy_evaluation_for_episode(
            episode)
        expected_values = [[0, 20], [30, 0]]
        expected_counts = [[0, 1], [1, 0]]
        for state in range(self.agent.num_states):
            for action in range(self.agent.num_actions):
                assert actual_values[state][action] == expected_values[state][action]
                assert actual_counts[state][action] == expected_counts[state][action]

    def test_episode_with_repeated_states(self):
        episode = [(0, 1, -5), (0, 0, 10), (1, 0, 0), (0, 0, 10)]
        actual_values, actual_counts = self.agent.policy_evaluation_for_episode(
            episode)
        expected_values = [[20, 15], [10, 0]]
        expected_counts = [[1, 1], [1, 0]]
        for state in range(self.agent.num_states):
            for action in range(self.agent.num_actions):
                assert actual_values[state][action] == expected_values[state][action]
                assert actual_counts[state][action] == expected_counts[state][action]


class TestPolicyEvaluation(unittest.TestCase):
    def setUp(self):
        num_states = 2
        num_actions = 2
        self.agent = MCAgent(2, 2, 1)

    def test_short_episode(self):
        episodes = [
            [(0, 1, -10), (1, 0, 30)],
            [(0, 1, -5), (0, 0, 10), (1, 0, 0), (0, 0, 10)],
            [(0, 0, 0), (0, 1, 10), (1, 0, -5), (1, 1, 20)]
        ]
        self.agent.policy_evaluation(episodes)
        expected_values = [[45/2, 20], [55/3, 20]]
        for state in range(self.agent.num_states):
            for action in range(self.agent.num_actions):
                assert self.agent.value_fn[state][action] == expected_values[state][action]

class TestArgMax(unittest.TestCase):
    def test_repeated_values(self):
        test_array = [0, 0, 0, 2, 0, 3, 0, 0, 1, 0]
        assert argmax(test_array) == 5, "Check your argmax implementation returns the index of the largest value"
        test_array = [0, 0, 0, 2, 0, 3, 0, 3, 1, 0]
        assert argmax(test_array) in [5, 7], "Check your argmax implementation returns the index of the largest value"
        test_array = [np.inf, -np.inf, 0.5, 0.5, 0.5]
        assert argmax(test_array) == 0, "Check your argmax implementation returns the index of the largest value"
        test_array = [-1.71669531, -1.69002165, -1.84892   , -1.71591288]
        assert argmax(test_array) == 1, "Check your argmax implementation returns the index of the largest value"

class TestStateRepFunc(unittest.TestCase):
    def test_state_rep(self):
        get_state_rep = get_state_rep_func((5,5))
        obs = (0, 0)
        state = get_state_rep(obs)
        assert state == 0

        obs = (0, 1)
        state = get_state_rep(obs)
        assert state == 5

        obs = (1, 0)
        state = get_state_rep(obs)
        assert state == 1

        obs = (4, 4)
        state = get_state_rep(obs)
        assert state == 24

        obs = 5
        state = get_state_rep(obs)
        assert state == 5
        
if __name__ == '__main__':
    unittest.main()
