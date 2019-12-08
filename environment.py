# from sequence_generator import Single_Signal_Generator
import numpy as np
import random
import pandas as pd
import alpha_vantage.timeseries as alpha

def get_ts(ticker, tick='batch_daily_adjusted', dropna=True):
    # tick can be 'intraday', 'daily_adjusted', 'batch_daily_adjusted'
    ts = alpha.TimeSeries(key='VVLO0LVEKD59PF4Y', output_format='pandas')
    if tick == 'intraday': 
        data, _ = ts.get_intraday(symbol=ticker,interval='1min', outputsize='full')
        data = data[['4. close']]
        data = data.rename(columns={"4. close": ticker})
    elif tick == 'daily_adjusted':
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        data = data[['5. adjusted close']]
        data = data.rename(columns={"5. adjusted close": ticker})
    elif tick == 'batch_daily_adjusted':
        data = []
        for s in ticker:
            df, _ = ts.get_daily_adjusted(symbol=s, outputsize='full')
            df = df[['5. adjusted close']]
            df = df.rename(columns={"5. adjusted close": s})
            data.append(df)
        data = pd.concat(data, axis=1, sort=True)
    if dropna:
        data.dropna(inplace=True)
    return data

# def find_ideal(p, just_once):
# 	if not just_once:
# 		diff = np.array(p[1:]) - np.array(p[:-1])
# 		return sum(np.maximum(np.zeros(diff.shape), diff))
# 	else:
# 		best = 0.
# 		i0_best = None
# 		for i in range(len(p)-1):
# 			best = max(best, max(p[i+1:]) - p[i])

# 		return best

class Market:

    def __init__(self, data, last_n_timesteps, trans_cost, risk_averse=0.2):
        
        self.no_of_actions = 5 # buy 50, buy 10, hold, sell 10, sell 50
        self.action_quantities = np.array([10000, 1000, 0, -1000, -10000])
        self.action_labels = ["buy 10000", "buy 1000", "hold", "sell 1000", "sell 1000"]
        self.initial_wealth = 1000000
        # self.cash_positions = np.zeros((data.shape[0]+1, 1)) # cash positions in $
        # self.cash_positions[0] = 1000000 # initial cash position
        self.data = data.values
        self.data_ewa = data.ewm(com=9.5).mean().values # 20-day exponentially moving average
        self.tickers = data.columns
        self.max_position = self.initial_wealth//np.min(self.data[0]) # max number of shares can hold given stock
        self.min_position = - self.max_position//2 # min number of shares can hold given stock
        self.positions = np.zeros(data.shape) # add initial position at t0, non-cash poositions are in number of shares
        self.valid_actions = self.get_actions() # list of possible actions e.g. [0,1,2,3,4]
        self.last_n_timesteps = last_n_timesteps
        self.trans_cost = trans_cost
        self.risk_averse = risk_averse
        self.state_shape = (last_n_timesteps, data.shape[1]) # # of stock price ts, position history
        self.portfolio_value = np.zeros((data.shape[0], 1)) # portfolio return, portfolio risk as states
        # self.portfolio_value[0] = 1000000 # initial portfolio value in cash
        self.start_index = last_n_timesteps - 1
        self.portfolio_value[:self.start_index+1] = self.initial_wealth
        self.current_index = self.start_index
        self.last_index = None
        self.reset()
	
    def reset(self):
        # self.isAvailable = True

        # data = self.data
        # sample_1d = np.reshape(data[:,0], data.shape[0])

        # self.data = data.copy()
        # self.normalized_values = sample_1d/sample_1d[0]*100
        rand_stock_idx = np.arange(self.data.shape[1])
        np.random.shuffle(rand_stock_idx)
        self.data = self.data[:, rand_stock_idx]
        self.data_ewa = self.data_ewa[:, rand_stock_idx]
        self.positions = self.positions[:, rand_stock_idx]
        self.tickers = self.tickers[rand_stock_idx]
        self.last_index = self.data.shape[0] - 1
        self.current_index = self.start_index

        return self.get_state()

    def get_state(self):
        # print(self.data.shape)
        # print(self.current_index)
        state = self.data_ewa[self.current_index - self.last_n_timesteps + 1: self.current_index + 1, :].copy()

        # normalize input
        for i in range(state.shape[1]):
            norm = np.mean(state[:,i])
            stdev = np.std(state[:,i])
            state[:,i] = (state[:,i] - norm)/stdev * 100

        return state

    def get_actions(self):
        # returns np nd array: no of possible actions x no of tickers
        # return np.array([np.arange(self.no_of_actions)] * len(self.tickers)).T 
        return np.arange(self.no_of_actions) 

    def get_trades(self, action):
        return np.array([self.action_quantities[action_for_stock_i] for action_for_stock_i in action])

    def get_reward(self, action):
        # calculate trades and update positions
        trades = self.get_trades(action) # get trades (list of change in shares for each stock)
        self.positions[self.current_index+1, :] = self.positions[self.current_index, :] + trades

        # prune trades based on trading limits
        self.positions[self.current_index+1, :] = np.clip(self.positions[self.current_index+1, :], self.min_position, self.max_position)
        # print('positions t+2', self.positions[self.current_index+2, :])
        trades_pruned = self.positions[self.current_index+1, :] - self.positions[self.current_index, :] # revised trades
        # print('trades_pruned', trades_pruned)
        # self.cash_positions[self.current_index+2, 0] = self.cash_positions[self.current_index+1, 0] + np.inner(trades_pruned * -1, self.data[self.current_index+1, :])

        # calculate reward (change in portfolio value)
        # self.portfolio_value[self.current_index+2, 0] = np.inner(self.data[self.current_index+1, :], self.positions[self.current_index+2, :]) - np.inner(self.data[self.current_index, :], self.positions[self.current_index+1, :]) # total asset mv change as reward
        # self.portfolio_value[self.current_index+2, 0] += self.cash_positions[self.current_index+2, 0] - self.cash_positions[self.current_index+1, 0] # adding change of cash s.t. reward reflect overall portfolio pnl
        # reward = self.portfolio_value[self.current_index+2, 0] - self.portfolio_value[self.current_index+1, 0] # reward is total portfolio pnl
        # reward -= np.sum(np.abs(trades_pruned)) * self.trans_cost         # apply transaction cost. assuming buy and sell have similar market impact

        # reward = (self.data[self.current_index+1, :] - self.data[self.current_index, :]) * self.positions[self.current_index, :] # reward by asset
        reward = (self.data_ewa[self.current_index+1, :] - self.data_ewa[self.current_index, :]) * self.positions[self.current_index, :] # reward by asset
        reward -= np.abs(trades_pruned) * self.trans_cost # reduce reward by transaction cost 

        # print('reward', reward)
        # print(self.portfolio_value[self.current_index+1, 0])
        self.portfolio_value[self.current_index+1, 0] = self.portfolio_value[self.current_index, 0] + np.sum((self.data[self.current_index+1, :] - self.data[self.current_index, :]) * self.positions[self.current_index, :])
        # self.portfolio_value[self.current_index+1, 0] = self.portfolio_value[self.current_index, 0] + np.sum(reward)
        return_since_inception = (self.portfolio_value[self.current_index+1, 0] / self.initial_wealth - 1) * 100
        if np.sum(reward) != 0.: reward = return_since_inception * reward / np.sum(reward) # pnl allocated to each asset
        else: reward = reward * 0
        # print(self.portfolio_value[self.current_index+2, 0])
        # trades_pruned * -1 * self.data[self.current_index+1, :] # cash change resulted from trading
        # reward += self.data[self.current_index+1, :], self.positions[self.current_index+2, :] - self.data[self.current_index, :], self.positions[self.current_index+1, :] # asset position mv change

        # assert len(reward) == len(self.tickers)

        # # add risk aversion
        reward = [r * (1. + self.risk_averse) if r<0 else r for r in reward ]
        # print(', price: ', self.data[self.current_index+1, :], 'trades: ', trades_pruned, ', positioins: ', self.positions[self.current_index+1, :], ', reward: ', reward, ', portfolio value: ', self.portfolio_value[self.current_index+1, :], '\n')

        return reward # normalize reward by t-1 portfolio value


    def step(self, action): # action is a list of chosen actions for each stock

        # if action == 0:		# don't buy / sell
        #     reward = 0.
        #     self.isAvailable = True
        # elif action == 1:	# buy
        #     reward = self.get_noncash_reward()
        #     self.isAvailable = False
        # elif action == 2:	# hold
        #     reward = self.get_noncash_reward()
        # else:
        #     raise ValueError('no such action: '+str(action))
        reward = self.get_reward(action) # reward at time t
        self.current_index += 1
        next_state = self.get_state()
        return next_state, reward, self.current_index == self.last_index

if __name__ == '__main__':
    # gen = Single_Signal_Generator(180, (10, 40), (5, 80), 0.5)
    ticker=['AAPL', 'IBM']
    sample = get_ts(ticker)
    # print(sample.shape)
    # print(sample.dtypes)
    env = Market(sample, 20, 3.3)
    env.reset()