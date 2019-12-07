# from sequence_generator import Single_Signal_Generator
import numpy as np
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

    def __init__(self, data, last_n_timesteps, buy_cost, risk_averse=0.05):
        
        self.no_of_actions = 5 # buy 50, buy 10, hold, sell 10, sell 50
        self.action_quantities = np.array([50, 10, 0, -10, -50])
        self.action_labels = ["buy 50", "buy 10", "hold", "sell 10", "sell 50"]
        self.max_position = 50 # max number of shares can hold given stock
        self.min_position = -20 # min number of shares can hold given stock
        self.positions = np.zeros((data.shape[0]+1, data.shape[1])) # add initial position at t0, non-cash poositions are in number of shares
        # self.cash_positions = np.zeros((data.shape[0]+1, 1)) # cash positions in $
        # self.cash_positions[0] = 1000000 # initial cash position
        self.data = data.values
        self.tickers = data.columns
        self.valid_actions = self.get_actions() # list of possible actions e.g. [0,1,2,3,4]
        self.last_n_timesteps = last_n_timesteps
        self.buy_cost = buy_cost
        self.risk_averse = risk_averse
        self.state_shape = (last_n_timesteps, data.shape[1]) # # of stock price ts, position history
        self.portfolio_value = np.zeros((data.shape[0]+1, 1)) # portfolio return, portfolio risk as states
        self.portfolio_value[0] = 1000000 # initial portfolio value in cash
        self.start_index = last_n_timesteps - 1
        self.current_index = self.start_index
        self.last_index = None
        self.reset()
	
    def reset(self):
        # self.isAvailable = True

        sample_2d = self.data
        # sample_1d = np.reshape(sample_2d[:,0], sample_2d.shape[0])

        self.sample_2d = sample_2d.copy()
        # self.normalized_values = sample_1d/sample_1d[0]*100
        self.last_index = self.sample_2d.shape[0] - 1

        # self.max_profit = find_ideal(self.normalized_values[self.start_index:], False)
        self.current_index = self.start_index

        return self.get_state()

    def get_state(self):
        state = self.sample_2d[self.current_index - self.last_n_timesteps + 1: self.current_index + 1, :].copy()

        for i in range(state.shape[1]):
            norm = np.mean(state[:,i])
            state[:,i] = (state[:,i]/norm - 1.)*100

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
        self.positions[self.current_index+2, :] = self.positions[self.current_index+1, :] + trades

        # prune trades based on trading limits
        self.positions[self.current_index+2, :] = np.clip(self.positions[self.current_index+2, :], self.min_position, self.max_position)
        # print('positions t+2', self.positions[self.current_index+2, :])
        trades_pruned = self.positions[self.current_index+2, :] - self.positions[self.current_index+1, :] # revised trades
        # print('trades_pruned', trades_pruned)
        # self.cash_positions[self.current_index+2, 0] = self.cash_positions[self.current_index+1, 0] + np.inner(trades_pruned * -1, self.sample_2d[self.current_index+1, :])

        # calculate reward (change in portfolio value)
        # self.portfolio_value[self.current_index+2, 0] = np.inner(self.sample_2d[self.current_index+1, :], self.positions[self.current_index+2, :]) - np.inner(self.sample_2d[self.current_index, :], self.positions[self.current_index+1, :]) # total asset mv change as reward
        # self.portfolio_value[self.current_index+2, 0] += self.cash_positions[self.current_index+2, 0] - self.cash_positions[self.current_index+1, 0] # adding change of cash s.t. reward reflect overall portfolio pnl
        # reward = self.portfolio_value[self.current_index+2, 0] - self.portfolio_value[self.current_index+1, 0] # reward is total portfolio pnl
        # reward -= np.sum(np.abs(trades_pruned)) * self.buy_cost         # apply transaction cost. assuming buy and sell have similar market impact

        reward = (self.sample_2d[self.current_index+1, :] - self.sample_2d[self.current_index, :]) * self.positions[self.current_index+1, :] # reward by asset
        # print('reward', reward)
        reward -= np.abs(trades_pruned) * self.buy_cost # reduce reward by transaction cost 
        # print('reward', reward)
        # print(self.portfolio_value[self.current_index+1, 0])
        self.portfolio_value[self.current_index+2, 0] = self.portfolio_value[self.current_index+1, 0] + np.sum(reward)
        # print(self.portfolio_value[self.current_index+2, 0])
        # trades_pruned * -1 * self.sample_2d[self.current_index+1, :] # cash change resulted from trading
        # reward += self.sample_2d[self.current_index+1, :], self.positions[self.current_index+2, :] - self.sample_2d[self.current_index, :], self.positions[self.current_index+1, :] # asset position mv change

        # assert len(reward) == len(self.tickers)

        # # add risk aversion
        # if reward < 0:
        #     reward *= (1. + self.risk_averse)

        return reward


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