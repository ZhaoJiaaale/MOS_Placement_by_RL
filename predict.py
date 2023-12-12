import standard_cell_layout
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
import json
from standard_cell_layout.envs.parseStdcell import get_stdcell_Graph
from stable_baselines3.common.callbacks import BaseCallback

def to_json(info, json_path):

    if info != {}:
        placement = {}
        layout = {}
        x = 0
        print(info)
        for i in range(len(info["layout"])):
            mos_pair = info["layout"][i]
            nmos = {}
            pmos = {}

            if mos_pair[0][-1].split("_")[0] == "Dummy":
                if mos_pair[1][1] == info["layout"][i-1][0][3]:
                    x += 1
                    pmos["x"] = str(x)
                else:
                    x += 2
                    pmos["x"] = str(x)

            elif mos_pair[1][-1].split("_")[0] == "Dummy":
                if mos_pair[0][1] == info["layout"][i-1][1][3]:
                    x += 1
                    nmos["x"] = str(x)
                else:
                    x += 2
                    nmos["x"] = str(x)

            
            elif mos_pair[0][2] == mos_pair[1][2]:
                # 共栅
                if mos_pair[0][1] == info["layout"][i-1][0][3] and mos_pair[1][1] == info["layout"][i-1][1][3]:
                    if i != 0:
                        x += 1
                else:
                    if i != 0:
                        x += 2
                nmos["x"] = str(x)
                pmos["x"] = str(x)
            
            else:
                # 不共栅
                if mos_pair[0][1] == info["layout"][i-1][0][3]:
                    if i != 0:
                        x += 1
                    nmos["x"] = str(x)
                    x += 1
                    pmos["x"] = str(x)
                elif mos_pair[1][1] == info["layout"][i-1][1][3]:
                    if i != 0:
                        x += 1
                    pmos["x"] = str(x)
                    x += 1
                    nmos["x"] = str(x)
                else:
                    if i != 0:
                        x += 2
                    nmos["x"] = str(x)
                    x += 1
                    pmos["x"] = str(x)

            if nmos != {}:
                nmos["y"] = mos_pair[0][0]
                nmos["source"] = mos_pair[0][1]
                nmos["gate"] = mos_pair[0][2]
                nmos["drain"] = mos_pair[0][3]
                nmos["width"] = str(mos_pair[0][4])
                nmos_name = mos_pair[0][-1]
                layout[nmos_name] = nmos

            if pmos != {}:
                pmos["y"] = mos_pair[1][0]
                pmos["source"] = mos_pair[1][1]
                pmos["gate"] = mos_pair[1][2]
                pmos["drain"] = mos_pair[1][3]
                pmos["width"] = str(mos_pair[1][4])
                pmos_name = mos_pair[1][-1]
                layout[pmos_name] = pmos

        placement["placement"] = layout
            
        with open(json_path, 'w') as json_file:
            json.dump(placement, json_file, indent=4)

cell_name = "AOI221D2"

env = gym.make("standard_cell_layout/StdCellPlaceEnv-v0")

best_save_path = "./model/best_Q/ppo_best_model_" + cell_name
last_save_path = "./model/last/ppo_last_model_" + cell_name

mos_num_name, nmos_list, pmos_list, s_g_d, mos_width = get_stdcell_Graph(cell_name)
num_episode = max(len(nmos_list), len(pmos_list))
json_path_best_Q = "./placement/best_Q/" + cell_name + ".json"
json_path_last = "./placement/last/" + cell_name + ".json"

def predict_(save_path, json_path, num_episode):
    loaded_model = PPO.load(save_path, env=env)
    vec_env = loaded_model.get_env()
    obs = vec_env.reset()
    done = False
    while not done:
        action, _states = loaded_model.predict(obs, deterministic=True)
        action = action[0] if len(action)==1 else action
        obs, reward, done, _, info = env.step(action)
        if info != {}:
            if len(info["layout"]) == num_episode:
                to_json(info, json_path)


predict_(best_save_path, json_path_best_Q, num_episode)
# predict_(last_save_path, json_path_last, num_episode)