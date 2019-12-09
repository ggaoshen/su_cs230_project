from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Simulator:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def play_one_episode(self, epsilon, training=True):

        self.agent.epsilon = epsilon

        state = self.env.reset()
        done = False
        prev_cum_reward = np.zeros(len(self.env.tickers))
        loss_history = []
        while not done: # step through time

            action = self.agent.act(state, self.env.valid_actions)
            next_state, reward, done = self.env.step(action)

            prev_cum_reward += reward

            if training:
                self.agent.remember((state, action, reward, next_state, done))
                loss = self.agent.replay(self.env.state_shape, self.env.action_shape_2d)
                loss_history.append(loss)

            state = next_state
        if training: return prev_cum_reward, loss_history[-1]
        else: return prev_cum_reward, loss_history

    def train(self, num_epochs, epsilon_decay=0.995, min_epsilon=0.01, epsilon=1, progress_report=100):

        exploration_episode_rewards = []
        episode_ending_portfolio_values = [] # portfolio value by each epoch 
        episode_ending_losses = []

        print("-"*60)
        print("Training")
        print("-"*60)

        for episode_no in tqdm(range(1, num_epochs+1)):
            exploration_episode_reward, episode_end_loss = self.play_one_episode(epsilon, training=True)
            exploration_episode_rewards.append(exploration_episode_reward)
            episode_ending_portfolio_values.append(self.env.portfolio_value[-1][0])
            episode_ending_losses.append(episode_end_loss)

            # print('\n', episode_no, exploration_episode_reward, self.env.portfolio_value[-1], self.env.positions[-1])

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay

        return episode_ending_portfolio_values, self.env.positions[:], exploration_episode_rewards, episode_ending_losses # return potf value by each episode, last episode's positions

    def test(self):

        print("-"*60)
        print("Testing")
        print("-"*60)

        for episode_no in tqdm(range(1)):
            test_episode_reward, _ = self.play_one_episode(0.2, training=False)
            
        return self.env.portfolio_value[:], self.env.positions[:], test_episode_reward # return potf value by each episode, last episode's positions