import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import pandas as pd
    
class Visualize():
    def __init__(self, env_train, env_test, ep_end_portf_val_train, exploration_episode_rewards_train, episode_ending_losses, run_details):
        self.env_train = env_train
        self.env_test = env_test
        self.ep_end_portf_val_train = ep_end_portf_val_train
        self.exploration_episode_rewards_train = exploration_episode_rewards_train
        self.episode_ending_losses = episode_ending_losses 
        self.run_details = run_details
        
    def save_test_results(self):
        
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(self.ep_end_portf_val_train)
        ax.set_xlabel('Epochs(episodes)')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Ending Portfolio Value by Training Epochs')
        fig.savefig('visualization/' + str(self.run_details) + '_1_end_portfval_by_epoch_'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png')
        
        training_reward = np.array(self.exploration_episode_rewards_train)
        fig, ax = plt.subplots(figsize=(12,8))
        ax = sns.distplot(np.sum(training_reward, axis=1), hist=False)
        ax.set_xlabel('Portfolio Value ($)')
        ax.set_title('Ending Total Reward by Training Epochs')
        fig.savefig('visualization/' + str(self.run_details) + '_2_end_totalreward_by_epoch_'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png')

        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(np.array(self.episode_ending_losses).reshape(-1))
        ax.set_xlabel('Epochs(episodes)')
        ax.set_title('Loss (MSE)')
        fig.savefig('visualization/' + str(self.run_details) + '_2A_end_MSEloss_by_epoch_'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png')
                
        buy_and_hold_strategy_portf_value, single_stock_strategy_portf_value = self.generate_benchmark_strats()
        
        # plot performance comparison
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(self.env_test.portfolio_value/self.env_test.portfolio_value[0] -1, label='dqn')
        ax.plot(buy_and_hold_strategy_portf_value/buy_and_hold_strategy_portf_value[0]-1, label='buy_and_hold')
        ax.plot(single_stock_strategy_portf_value/single_stock_strategy_portf_value[0]-1, label='best_single_stock')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Wealth Appreciation')
        ax.set_title('Test DQN Strategy Against Benchmarks by Test Time Steps')
        ax.legend()
        fig.savefig('visualization/' + str(self.run_details) + '_3_test_dqn_vs_bm_over_time_'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png')
        
        fig, ax = self.plot_positions(self.env_train, long_short='long', title='Long Positions - Train')
        fig.savefig('visualization/' + str(self.run_details) + '_4_train_positions_'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png')
        
        fig, ax = self.plot_positions(self.env_train, long_short='short', title='Short Positions - Train')
        fig.savefig('visualization/' + str(self.run_details) + '_5_train_positions_'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png')

        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(self.env_train.cash_positions/self.env_train.portfolio_value[0] - 1)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Cash Positions')
        ax.set_title('Cash Positions by Training Time Steps')
        fig.savefig('visualization/' + str(self.run_details) + '_5A_train_cashpos_'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png')
        
        fig, ax = self.plot_positions(self.env_test, long_short='long', title='Long Positions - Test')
        fig.savefig('visualization/' + str(self.run_details) + '_6_test_positions_'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png')
        
        fig, ax = self.plot_positions(self.env_test, long_short='short', title='Short Positions - Test')
        fig.savefig('visualization/' + str(self.run_details) + '_7_test_positions_'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png')

    def generate_benchmark_strats(self):
        # create buy and hold strategy for test period
        buy_and_hold_strategy_portf_value = np.array([ np.sum((self.env_test.portfolio_value[0] / self.env_test.data.shape[1]) / self.env_test.data[0] * self.env_test.data[i]) for i in range(1, self.env_test.data.shape[0])])

        # create best single stock strategy
        best_stock_idx = np.argmax(self.env_test.data[-1]/self.env_test.data[0] - 1)
        print('best stock pick: ', self.env_test.tickers[best_stock_idx])
        single_stock_strategy_portf_value = self.env_test.portfolio_value[0]/self.env_test.data[0, best_stock_idx]*self.env_test.data[:, best_stock_idx]
        single_stock_strategy_portf_value
        return buy_and_hold_strategy_portf_value, single_stock_strategy_portf_value
        
    def plot_positions(self, environ, long_short='long', title='Long Positions'): 
        if long_short=='long': 
            d=np.clip(environ.positions, 0, None)
            df = pd.DataFrame(d, columns=environ.tickers.values)
        else: 
            d=np.clip(environ.positions, None, 0)
            df = pd.DataFrame(d, columns=environ.tickers.values)
        # We need to transform the data from raw data to percentage (fraction)
        # plt.stackplot(df.index, df.T)

        fig, ax = plt.subplots(figsize=(12,8))
        ax.stackplot(df.index, df.T, labels=environ.tickers.values)
        ax.legend(loc='best')
        ax.set_title(title)
        ax.set_ylabel('No. of Shares')
        ax.set_xlabel('Time Steps')
        plt.show()
        return fig, ax