
import random

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

import numpy as np
import gymnasium as gym
import pandas as pd
# import seaborn as sns

from TestEnv import Electric_Car

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from mpl_toolkits import mplot3d
from matplotlib import cm
import pandas as pd
import smart_grid
from TestEnv import Electric_Car 
import seaborn as sns
import math



num_inputs = 4
num_actions = 2
layer = tf.keras.layers.Normalization(axis=None)
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu",input_shape=(num_inputs,)),
    # keras.layers.Dropout(0.02, noise_shape=None, seed=None),
    keras.layers.Dense(num_actions, activation="softmax")
])

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01))

def run_episode(max_steps_per_episode = 10000,render=False):    
    states, actions, probs, rewards ,ac_rew= [],[],[],[],[]
    # env=Electric_Car('./validate.xlsx')
    # fname='./validate.xlsx'
    # state = reset_env(env,fname)
    # print('initial state',state)
    done=False
    state=env.reset()
    # print(state[0])
    # print('obs',obs)

    # l1=list(obs)
    # l1.append(0.0)
    # state=np.asarray(l1)
    state=state[0]
    while not done:

        arr=np.expand_dims(state,0)
        action_probs = model(arr)
        action = np.random.choice(num_actions, p=np.squeeze(action_probs))
        nstate, reward, terminated, truncated, _ = env.step(action)
        env.render()


        done=terminated
        if done:
            break
        states.append(state)
        actions.append(action)
        probs.append(action_probs)
        rewards.append(reward)
        state = nstate
    return np.vstack(states), np.vstack(actions), np.vstack(probs), np.vstack(rewards)


s,a,p,r= run_episode()
print(f"Total reward: {np.sum(r)}")
eps = 0.001



def discounted_rewards(rewards,gamma=0.99,normalize=False):
    ret = []
    s = 0
    for r in rewards[::-1]:
        s = r + gamma * s
        ret.insert(0, s)
    if normalize:
        ret = (ret-np.mean(ret))/(np.std(ret)+eps)
    return ret

alpha = 1e-4

history = []
for epoch in range(1000):
    states, actions, probs, rewards = run_episode()
    # print('actions',actions)
    one_hot_actions = np.eye(2)[actions.T][0]
    gradients = one_hot_actions-probs
    dr = discounted_rewards(rewards)
    # print(dr)
    gradients *= dr
    target = alpha*np.vstack([gradients])+probs
    model.train_on_batch(states,target)
    history.append(np.sum(rewards))
    # if epoch%100==0:
    print(f"{epoch} -> {np.sum(rewards)}")
model.save_weights('agent2.h5')

model.save('./agent2.keras')

# play_validation_game(model)
plt.plot(history)
plt.savefig('./agent.png')

# _ = run_episode(render=True)

