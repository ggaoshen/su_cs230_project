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
        # print(prev_cum_reward)
        while not done:

            action = self.agent.act(state, self.env.valid_actions)
            next_state, reward, done = self.env.step(action)

            prev_cum_reward += reward
            # print(prev_cum_reward)

            if training:
                self.agent.remember((state, action, reward, next_state, done))
                self.agent.replay()

            state = next_state

        return prev_cum_reward

    def train(self, num_epochs, epsilon_decay=0.995, min_epsilon=0.01, epsilon=1, progress_report=100):

        exploration_episode_rewards = []
        episode_ending_portfolio_values = [] # portfolio value by each epoch 
        # exploration_max_episode_rewards = []
        # safe_max_episode_rewards = []

        print("-"*60)
        print("Training")
        print("-"*60)

        for episode_no in tqdm(range(1, num_epochs+1)):

            # exploration_episode_reward, exploration_max_episode_reward  = self.play_one_episode(epsilon, training=True)
            # exploration_episode_rewards.append(exploration_episode_reward)
            # exploration_max_episode_rewards.append(exploration_max_episode_reward)
            exploration_episode_reward = self.play_one_episode(epsilon, training=True)
            exploration_episode_rewards.append(exploration_episode_reward)
            episode_ending_portfolio_values.append(self.env.portfolio_value[-1][0])

            # print('\n', episode_no, exploration_episode_reward, self.env.portfolio_value[-1], self.env.positions[-1])

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay

            # if episode_no % progress_report == 0:
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(2, 1, 1)
            #     ax1.plot(exploration_episode_rewards, 'blue')
            #     # ax1.plot(exploration_max_episode_rewards, 'blue')

            #     # ax2 = fig.add_subplot(2, 1, 2)
            #     # ax2.plot(safe_episode_rewards, 'blue')
            #     # ax2.plot(safe_max_episode_rewards, 'blue')

            #     fig.savefig('training_progress_' + str(episode_no) + '_episodes.png')

        # fig = plt.figure()
        # ax1 = fig.add_subplot(2, 1, 1)
        # ax1.plot(episode_ending_portfolio_values, 'blue')
        # ax2 = fig.add_subplot(2, 1, 2)
        # ax2.plot(self.env.positions[:])
        # plt.savefig('training_progress_and_positions_' + str(num_epochs) + '_episodes.png')

        return episode_ending_portfolio_values, self.env.positions[:], exploration_episode_rewards # return potf value by each episode, last episode's positions

    def test(self):

        # test_episode_rewards = []
        # episode_ending_portfolio_values = [] # portfolio value by each epoch 

        print("-"*60)
        print("Testing")
        print("-"*60)

        for episode_no in tqdm(range(1)):

            # test_episode_reward, max_reward = self.play_one_episode(0, training=False)
            # test_episode_rewards.append(test_episode_reward)
            # max_rewards.append(max_reward)
            test_episode_reward = self.play_one_episode(0.2, training=False)
            # test_episode_rewards.append(test_episode_reward)
            # episode_ending_portfolio_values.append(self.env.portfolio_value[-1])
            # max_rewards.append(max_reward)
            
            # print('\n', episode_no, test_episode_reward, self.env.portfolio_value[-1], self.env.positions[-1])
            # print('\n test positions shape', self.env.positions.shape)
            # print('\n test test_episode_reward shape', test_episode_reward.shape)
        
        # fig = plt.figure()
        # ax1 = fig.add_subplot(2, 1, 1)
        # ax1.plot(self.env.portfolio_value[:], 'blue')

        # ax2 = fig.add_subplot(2, 1, 2)
        # ax2.plot(self.env.positions[:])

        # plt.savefig('test_results_and_positions.png')

        # positive_percentage = sum(x > 0 for x in test_episode_rewards)/len(test_episode_rewards)

        # print("-"*60)
        # print("Mean Reward:", np.mean(test_episode_rewards))
        # # print("Mean Max Reward:", np.mean(max_rewards))
        # # print("Positive Reward Percentage:", positive_percentage)
        # print("-"*60)

        return self.env.portfolio_value[:], self.env.positions[:], test_episode_reward # return potf value by each episode, last episode's positions