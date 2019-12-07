from environment import Market, get_ts
from model import Q_Model
from agent import Agent
from simulator import Simulator
import pandas as pd

# sampler = Single_Signal_Generator(total_timesteps=180, period_range=(10, 40), amplitude_range=(5, 80), noise_amplitude_ratio=0.5)
# filename = "Generated Signals.npy"
# sampler.build_signals(filename, 1000)
# sampler.load(filename)
# env = Market(sampler=sampler, last_n_timesteps=40, buy_cost=3.3)

# ticker=['AAPL', 'TSLA']
# # ticker=['TSLA']
# sample = get_ts(ticker)
# sample.to_csv('data/test/test_data.csv')
# import os
# print(os.getcwd())
sample = pd.read_csv('data/test/test_data.csv', index_col=0)
print(sample.head())
# print(sample.shape)
# print(sample.dtypes)
env_train = Market(sample.iloc[:1500, :], 20, 0.5)
env_test = Market(sample.iloc[1500:, :], 20, 0.5)
# env.reset()

dense_model = [
    {"type":"Reshape", "target_shape":(env_train.get_state().shape[0]*env_train.get_state().shape[1],)},
    {"type":"Dense", "units":128*env_train.get_state().shape[1]},
    {"type":"Dense", "units":32}
]
conv_model = [
    {"type":"Reshape", "target_shape":env_train.get_state().shape},
    {"type":"Conv1D", "filters":16, "kernel_size":3, "activation":"relu"},
    {"type":"Conv1D", "filters":16, "kernel_size":3, "activation":"relu"},
    {"type":"Flatten"},
    {"type":"Dense", "units":48, "activation":"relu"},
    {"type":"Dense", "units":24, "activation":"relu"}
]
gru_model = [
    {"type":"Reshape", "target_shape":env_train.get_state().shape},
    {"type":"GRU", "units":16, "return_sequences":True},
    {"type":"GRU", "units":16, "return_sequences":False},
    {"type":"Dense", "units":16, "activation":"relu"},
    # {"type":"Dense", "units":16, "activation":"relu"}
]
lstm_model = [
    {"type":"Reshape", "target_shape":env_train.get_state().shape},
    {"type":"LSTM", "units":16, "return_sequences":True},
    {"type":"LSTM", "units":16, "return_sequences":False},
    {"type":"Dense", "units":16, "activation":"relu"},
    # {"type":"Dense", "units":16, "activation":"relu"}
]

q_model = Q_Model("Dense", state_dim=env_train.get_state().shape, no_of_actions=env_train.no_of_actions, layers=dense_model, hyperparameters={"lr":0.0001})
agent = Agent(q_model, batch_size=8, discount_factor=0.995, epsilon=0.5)

no_epochs = 1
no_of_episodes_test = 1

sim = Simulator(env_train, agent)
train_rewards = sim.train(no_epochs, epsilon_decay=0.997)
agent.model.save()

sim_test = Simulator(env_test, agent)
test_rewards = sim_test.test()