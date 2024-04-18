import torch
import numpy as np

class BanditDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, rewards):
        assert states.shape[0] == actions.shape[0] and states.shape[0] == rewards.shape[0]
        assert len(states.shape) == 2
        assert len(actions.shape) == 1 or len(actions.shape) == 2
        assert len(rewards.shape) == 1

        self.length = actions.shape[0]

        self.states = torch.from_numpy(states).float()
        if torch.is_tensor(actions):
            self.actions = actions.float()
        elif isinstance(actions, np.ndarray):
            self.actions = torch.from_numpy(actions).float()
        else:
            raise ValueError()
        self.rewards = torch.from_numpy(rewards).float()

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        reward = self.rewards[index]

        return state, action, reward
    

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, rewards):
        assert states.shape[0] == actions.shape[0] and states.shape[0] == rewards.shape[0]
        assert len(states.shape) == 1
        assert len(actions.shape) == 1 or len(actions.shape) == 2
        assert len(rewards.shape) == 1

        self.length = actions.shape[0]

        self.states = torch.from_numpy(states)
        if torch.is_tensor(actions):
            self.actions = actions
        elif isinstance(actions, np.ndarray):
            self.actions = torch.from_numpy(actions)
        else:
            raise ValueError()
        self.rewards = torch.from_numpy(rewards).float()

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index].unsqueeze(0)
        reward = self.rewards[index]

        return state, action, reward