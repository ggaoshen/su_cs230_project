import numpy as np
import random

class Agent:

    def __init__(self, model, batch_size, discount_factor, epsilon):
        self.model = model
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.memory = []
        self.epsilon = epsilon

    def _get_q_valid(self, state):
        q_valid = self.model.predict(state)[0]
        # q_valid = [np.nan] * len(q)
        # for action in valid_actions:
        #     q_valid[action] = q[action]
        return q_valid.reshape((len(q_valid)//state.shape[1], state.shape[1]))

    def act(self, state, valid_actions):
        if np.random.random() > self.epsilon:
            q_valid = self._get_q_valid(state)
            # if np.nanmin(q_valid) != np.nanmax(q_valid):
            return np.nanargmax(q_valid, axis=0)
        return np.array(random.sample(list(valid_actions), state.shape[1]))

    def remember(self, experience):
        self.memory.append(experience)

    def replay(self):
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in batch:
            q = reward
            if not done:
                q += self.discount_factor * np.nanmax(self._get_q_valid(next_state), axis=0)
            self.model.fit(state, action, q)

    def save_model(self):
        self.model.save()
