from environment import Market, get_ts
from model import Q_Model
from agent import Agent
from simulator import Simulator
from visualization import Visualize
import pandas as pd
import matplotlib.pyplot as plt

# ticker=['MS']
# ticker=['TSLA']
# ticker=['TSLA', 'GOOGL', 'MS', 'KO', 'CVX', 'HD', 'MMM' ,'BA']
# sample = get_ts(ticker)
# sample.to_csv('data/test/test_data_8stockmix.csv')
sample = pd.read_csv('data/test/test_data_8stockmix.csv', index_col=0)
# sample = pd.read_csv('data/test/test_data.csv', index_col=0)
print(sample.head())
env_train = Market(sample.iloc[:1500, :], 20, 0.5)
env_test = Market(sample.iloc[1500:, :], 20, 0.5)

dense_model = [
    {"type":"Reshape", "target_shape":(env_train.get_state().shape[0]*env_train.get_state().shape[1],)},
    {"type":"Dense", "units":64*env_train.get_state().shape[1]},
    {"type":"Dense", "units":64}
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
]

# user input
no_epochs = 250
model_name = 'Dense_8stock_longonly_ewa'
run_details = model_name + '_' + str(no_epochs) + '_eps'
learning_rate = 0.0001

# build model and agent
q_model = Q_Model(model_name, state_dim=env_train.get_state().shape, no_of_actions=env_train.no_of_actions, layers=dense_model, hyperparameters={"lr":learning_rate})
agent = Agent(q_model, batch_size=8, discount_factor=0.995, epsilon=0.7)

# train and test
sim = Simulator(env_train, agent)
ep_end_portf_val_train, last_eps_position_ts_train, exploration_episode_rewards_train, episode_ending_losses = sim.train(no_epochs, epsilon_decay=0.995)

# save model snapshot 
agent.model.save() 

sim_test = Simulator(env_test, agent)
test_portf_val_ts, test_position_ts, test_ending_rewards = sim_test.test()

# visualize result
viz = Visualize(env_train, env_test, ep_end_portf_val_train, exploration_episode_rewards_train, episode_ending_losses, run_details)
viz.save_test_results()