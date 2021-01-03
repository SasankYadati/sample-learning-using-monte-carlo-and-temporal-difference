import random

class Agent():
    def __init__(self, num_states, num_actions, alpha, discount_rate=1.0):
        """
        One agent to base them all.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.discount_rate = discount_rate
        self.value_fn = [0] * self.num_states
        self.policy = []
        for _ in range(self.num_states):
            p_s = [0] * self.num_actions
            random_action = random.randint(0, self.num_actions-1)
            p_s[random_action] = 1
            self.policy.append(p_s)