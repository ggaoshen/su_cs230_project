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
        q_valid = self.model.predict(state.reshape((1,)+state.shape))[0]
        # q_valid = [np.nan] * len(q)
        # for action in valid_actions:
        #     q_valid[action] = q[action]
        # print(q_valid.shape)
        return q_valid.reshape((len(q_valid)//state.shape[1], state.shape[1]))

    def act(self, state, valid_actions):
        if np.random.random() > self.epsilon:
            q_valid = self._get_q_valid(state)
            # if np.nanmin(q_valid) != np.nanmax(q_valid):
            return np.nanargmax(q_valid, axis=0)
        # return np.array(random.sample(list(valid_actions), state.shape[1]))
        return np.random.choice(len(valid_actions), state.shape[1])

    def remember(self, experience):
        self.memory.append(experience)

    def replay(self, state_size, action_size_2d):
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        X = np.empty((len(batch), )+state_size) # (no of replay steps, no of price lookback window, no of stocks) 
        Y = np.empty((len(batch), )+action_size_2d) # (no of replay steps, no of valid actions, no of stocks) 
        # q_hat_2d = self._get_q_valid(state)
        
        for i in range(len(batch)): 
            state, action, reward, next_state, done = batch[i]
            q_hat_2d = self._get_q_valid(state) # reshape q_hat: self.no_of_actions X self.state_dim[1]
            q = reward
            if not done:
                q_hat_new_2d = self._get_q_valid(next_state)
                q += self.discount_factor * np.nanmax(q_hat_new_2d, axis=0)

            for stock_i, action_for_stock_i in enumerate(action):
                q_hat_2d[action_for_stock_i, stock_i] = q[stock_i]

            X[i] = state
            Y[i] = q_hat_2d
        # self.model.fit(X, Y)
        return self.model.fit(X, Y)

    def save_model(self):
        self.model.save()
