import os, sys
import numpy as np
from smac.env import StarCraft2Env
from model import DGN
from buffer import ReplayBuffer
from config import *
from utilis import *
import torch
import torch.optim as optim

test_env = StarCraft2Env(map_name='25m')
env_info = test_env.get_env_info()
n_ant = env_info["n_agents"]
n_actions = env_info["n_actions"]
obs_space = env_info["obs_shape"] + n_ant

model = DGN(n_ant,obs_space,hidden_dim,n_actions).cuda()
task_path = os.path.dirname(os.path.realpath(__file__))
load_path = task_path + "/Weights/25/full_" + str(482139) + ".pt"
model.load_state_dict(torch.load(load_path)["actor_architecture_state_dict"])

test_r, test_win = 0, 0
for _ in range(20):
    test_env.reset()
    test_obs = get_obs(test_env.get_obs(),n_ant)
    test_adj = test_env.get_visibility_matrix()[:,0:n_ant]*1 + np.eye(n_ant)
    test_mask = np.array([test_env.get_avail_agent_actions(i) for i in range(n_ant)])
    terminated = False
    while terminated == False:
        test_env.render()
        time.sleep(0.05)
        action=[]
        q = model(torch.Tensor(np.array([test_obs])).cuda(), torch.Tensor(np.array([test_adj])).cuda())[0]
        for i in range(n_ant):
            a = np.argmax(q[i].cpu().detach().numpy() - 9e15*(1 - test_mask[i]))
            action.append(a)
        reward, terminated, winner = test_env.step(action)
        test_r += reward
        if winner.get('battle_won') == True:
            test_win += 1
        test_obs = get_obs(test_env.get_obs(),n_ant)
        test_adj = test_env.get_visibility_matrix()[:,0:n_ant]*1 + np.eye(n_ant)
        test_mask = np.array([test_env.get_avail_agent_actions(i) for i in range(n_ant)])

print(test_r/20, test_win/20)
