#it might print some test subs because i created this in j

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
%matplotlib inline
def running_mean(data,window=50):
    c = data.shape[0] - window
    smoothened = np.zeros(c)
    conv = np.ones(window)
    for i in range(c):
        smoothened[i] = (data[i:i+window] @ conv)/window
    return smoothened

def one_hot_encode(pos, dim):
    vec = np.zeros(dim)
    vec[pos] = 1
    return vec

print(one_hot_encode(0,4))
print(one_hot_encode(1,4))
print(one_hot_encode(2,4))
print(one_hot_encode(3,4))

def softmax(data, tau=1.2):
    softm = np.exp(data/tau) / np.sum(np.exp(data/tau))
    return softm


class Environment(object):

    def __init__(self, arms):
        self.arms = arms 
        self.reward_probas = np.random.rand(arms, arms)
        self._update_state()

    def _update_state(self):
        self.state = np.random.randint(0, self.arms)

    def get_state(self):
        return self.state 

    def _get_reward(self, arm):
        state = self.get_state()
        prob = self.reward_probas[state][arm]
        rewards = [1 if np.random.random() < prob else 0 for _ in range(self.arms)]
        return sum(rewards)

    def choose_arm(self, arm):
        reward = self._get_reward(arm)
        self._update_state()
        return reward

    
env = Environment(arms=5)
print(env.reward_probas)
state = env.get_state()
print(state)
env.choose_arm(2)
print(env.get_state())
n_arms = 10
n_actions = 10
model = nn.Sequential(
    nn.Linear(n_arms, 100),
    nn.ReLU(),
    nn.Linear(100, n_actions),
    nn.ReLU()
)

print(model)

def train_network(environ, net, epochs=8000, lr=1e-2):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    rewards = []
    losses = []
    for e in range(1, epochs + 1):
        state = torch.Tensor(one_hot_encode(environ.get_state(), dim=n_arms))
        rewards_pred = net(state)
        action_probas = softmax(rewards_pred.data.numpy().copy())

        arm = np.random.choice(n_arms, p=action_probas)
        reward = environ.choose_arm(arm)
        rewards.append(reward)

        true_rewards = rewards_pred.data.numpy().copy()
        true_rewards[arm] = reward 

        loss = criterion(rewards_pred, torch.Tensor(true_rewards))
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (np.array(losses), np.array(rewards))


env = Environment(arms=n_arms)
print(f"CURRENT STATE : {env.get_state()}")
print(env.reward_probas)

losses, rewards = train_network(env, model)

plt.plot(running_mean(rewards, window=500), label="avg reward")
plt.legend()

state = torch.Tensor(one_hot_encode(5, dim=10))
preds = model(state)
print(preds)
