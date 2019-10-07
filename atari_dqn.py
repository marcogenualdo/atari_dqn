from copy import copy
from itertools import cycle
import random

from threading import Thread
from queue import Queue

import numpy as np
import torch
from torch import nn
import gym

from models import DQN, Memory


### PARAMETERS ### 
game = 'Pong-v0' 
frames_number = 3
frames_to_skip = 2
gamma = 0.99

win_baseline = 15
wins_to_quit = 50

batch_size = 32
target_update_time = 1000
head_start = 1000
max_training_queue_size = 500

# process function: 
# - cuts the top and bottom of every frame, 
# - skips half of the pixels, 
# - turns the image to grayscale,
# - converts it to type 'short unsigned int' for cheap storage.

process_frame = lambda screen: \
    screen[32:-16:2,::2].mean(axis=2).astype(np.uint8)


### SETUP ###
env = gym.make(game)
win_streak = []
frame_shape = process_frame(env.reset()).shape
state_shape = (frames_number, *frame_shape)

if not torch.cuda.is_available(): print('cuda not available')
device = torch.device('cuda')

net = DQN(state_shape, env.action_space.n).to(device)
target_net = copy(net).to(device)

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

training_queue = Queue()
memory = Memory(int(1e+4), training_queue, device)


### UTILITY FUNCTIONS ###
def step (action, reset=False):
    if reset:
        state = [process_frame(env.reset())]
        loops = frames_number - 1
    else:
        state = []
        loops = frames_number

    reward = 0.
    for k in range(loops):
        for j in range(frames_to_skip):
            frame, r, done, info = env.step(action)
            reward += r
        state.append(process_frame(frame))

    state = np.stack(state)
    return state, reward, done, info


def eps_greedy_action (eps, state):
    if random.random() < eps:
        return env.action_space.sample()

    net_input = torch.tensor(state, dtype=torch.uint8, device=device).unsqueeze(0)
    return net(net_input).argmax().item()


def update_greed (episode):
    return max(0.05, 1. - episode / 1000)
    

def wait_for_training ():
    while training_queue.qsize() > 5:
        pass


def fill_queue ():
    while play_thread.is_alive():
        memory.sample_and_enqueue(batch_size)
        
        if training_queue.qsize() > max_training_queue_size:
            wait_for_training()


def stopper (episode_reward):
    if episode_reward > win_baseline:
        win_streak.append(episode_reward)
    else:
        win_streak.clear()

    if len(win_streak) > wins_to_quit:
        return True
    return False


### training loop ###
def train ():
    while play_thread.is_alive():
        if not training_queue.empty():
            (old_states, old_actions, rewards, new_states, mask) = training_queue.get()

            with torch.no_grad():
                target = rewards + gamma * mask * target_net(new_states).max(1)[0]

            optimizer.zero_grad()
            loss(target, net(old_states).gather(1, old_actions).flatten()).backward()
            optimizer.step()

    #NOTE: .flatten() after gather is very important, without it the agent does not learn anything! It's some quirk inside torch's MSELoss. It took me a day to figure out


### playing loop ###
def play ():
    target_update_loop = cycle(range(target_update_time))
 
    episode = 1
    won_the_game = False
    while not won_the_game:
        eps = update_greed(episode)

        state, reward, done, info = step(env.action_space.sample(), True)
        action = eps_greedy_action(eps, state)

        memory.remember(state, action, reward, done)

        episode_reward = 0.
        done = False
        while not done:
            state, reward, done, info = step(action)
            action = eps_greedy_action(eps, state)

            memory.remember(state, action, reward, done)

            if not next(target_update_loop):
                target_net.load_state_dict(net.state_dict())

            episode_reward += reward

        print(f'Episode: {episode}, Score = {episode_reward}, Queue size = {training_queue.qsize()}')
        
        won_the_game = stopper(episode_reward)
        episode += 1

        if training_queue.qsize() > max_training_queue_size:
            wait_for_training()



### EXECUTION ###
play_thread = Thread(target=play)
queueing_thread = Thread(target=fill_queue)

play_thread.start()

# head start
while len(memory.data) < head_start:
    pass 
print('Starting Training')

queueing_thread.start()
train()
torch.save(net.state_dict(), 'atari_model')
