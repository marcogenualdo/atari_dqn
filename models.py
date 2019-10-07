from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        print(f'{input_shape}, Neurons in first Linear layer: {conv_out_size}')

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


### memory ###
class Memory:
    event = namedtuple(
        'event', 
        ('state', 'action', 'reward', 'done')
    )

    def __init__ (self, size, out_queue, out_device):
        self.size = size
        self.data = deque()
        self.out_queue = out_queue
        self.out_device = out_device


    def remember (self, state, action, reward, done):
        self.data.append(self.event(
            state,
            action,
            reward,
            done
        ))

        if len(self.data) > self.size:
            self.data.popleft()


    def sample_and_enqueue (self, n):
        if len(self.data) < 2 * n: return

        start_indices = torch.randint(len(self.data) - 1, (n,))  
       
        old_states, new_states, actions, rewards, mask = [], [], [], [], []
        for start in start_indices:
            old_states.append(self.data[start].state)
            new_states.append(self.data[start + 1].state)
            actions.append(self.data[start].action)
            rewards.append(self.data[start].reward)
            mask.append(0. if self.data[start].done else 1.)

        old_states = np.stack(old_states)
        new_states = np.stack(new_states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        mask = np.stack(mask)

        self.out_queue.put((
            torch.tensor(old_states, dtype=torch.uint8, device=self.out_device),
            torch.tensor(actions, dtype=torch.int64, device=self.out_device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=self.out_device),
            torch.tensor(new_states, dtype=torch.uint8, device=self.out_device),
            torch.tensor(mask, dtype=torch.float32, device=self.out_device)
        ))
