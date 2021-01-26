import numpy as np
import random
from utils import argmax

class SarsaAgent():
    """
    A model free agent that learns from samples by bootstrapping.
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