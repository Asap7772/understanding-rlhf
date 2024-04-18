import numpy as np
import torch


# regular reward, all positive, 1d continuous
def reward_1(context, action):
    if torch.is_tensor(context):
        context = context.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
        
    reward = np.exp(-(np.linalg.norm(action - 0.7)/0.75) ** 2)
    return reward


# all positive, but more peaky, 1d continuous
def reward_2(context, action):
    if torch.is_tensor(context):
        context = context.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
        
    reward = np.exp(-(np.linalg.norm(action - 0.7)/0.25) ** 2)
    return reward


# all positive, but reward aligned with pi_data, 1d continuous
def reward_3(context, action):
    if torch.is_tensor(context):
        context = context.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
        
    reward = np.exp(-(np.linalg.norm(action + 0.7)/0.25) ** 2)
    return reward


# reward for the 1d discrete setting, not aligned
def reward_4(context, action):
    if torch.is_tensor(context):
        context = context.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()

    reward = np.exp(-(np.linalg.norm(action - 70.0)/10.0) ** 2)
    return reward


# reward for the 1d discrete setting, aligned
def reward_5(context, action):
    if torch.is_tensor(context):
        context = context.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()

    reward = np.exp(-(np.linalg.norm(action - 20.0)/10.0) ** 2)
    return reward


# reward for the N-d transformer discrete setting, not aligned
def reward_6(context, action):
    if torch.is_tensor(context):
        context = context.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
        
    reward = 0.0
    for i in range(action.shape[0]):
        reward += np.exp(-((action[i] - 70.0)/10) ** 2)
    reward /= action.shape[0]
    return reward


# reward for the N-d transformer discrete setting, aligned
def reward_7(context, action):
    if torch.is_tensor(context):
        context = context.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
        
    reward = 0.0
    for i in range(action.shape[0]):
        reward += np.exp(-((action[i] - 20.0)/10) ** 2)
    reward /= action.shape[0]
    return reward


# reward for the 1d discrete setting, not aligned, more spread out
def reward_8(context, action):
    if torch.is_tensor(context):
        context = context.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()

    reward = np.exp(-(np.linalg.norm(action - 70.0)/20.0) ** 2)
    return reward


# reward for the 1d discrete setting, not aligned, most spread out
def reward_9(context, action):
    if torch.is_tensor(context):
        context = context.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()

    reward = np.exp(-(np.linalg.norm(action - 70.0)/30.0) ** 2)
    return reward
    

def choose_reward_function(args):
    reward_function_id = args["reward_function"]
    reward_functions_dict = {
        1: reward_1,
        2: reward_2,
        3: reward_3,
        4: reward_4,
        5: reward_5,
        6: reward_6,
        7: reward_7,
        8: reward_8,
        9: reward_9,
    }

    if reward_function_id not in reward_functions_dict:
        raise ValueError(
            f"Given reward function id {reward_function_id} not supported."
        )
    
    return reward_functions_dict[reward_function_id]