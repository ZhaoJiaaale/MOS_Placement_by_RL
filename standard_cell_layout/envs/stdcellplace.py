import gymnasium as gym
from gymnasium import spaces
import numpy as np
from standard_cell_layout.envs.parseStdcell import get_stdcell_Graph
import json

class StdCellPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # 初始化

        self.stdcell_name = "AOI221D2"
        mos_num_name, nmos_list, pmos_list, s_g_d, mos_width = get_stdcell_Graph(self.stdcell_name)
        self.num_mos = len(nmos_list) + len(pmos_list)

        self.num_of_episode = max(len(nmos_list), len(pmos_list))

        self.count = 0
        self.chosen_mos = set()
        self.mos_num_name = mos_num_name
        self.placed_mos_pair = []
        self.not_share_gate_count = 0
        self.Q = 0
        self.s_g_d = s_g_d
        self.mos_width = mos_width
        self.info = {}
        self.offset = int(self.num_mos/2)
        self.observation = np.array([[0 for i in range(self.num_mos)], 
                                     [0 for i in range(self.num_mos)]])
        self.nets = []

        """
            dimension 1: Whether this mos is placed.
            dimension 2: Whether this mos is flipped.
        """
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=[2, self.num_mos],
            dtype=int
        )

        self.action_space = spaces.MultiDiscrete(
            [int(self.num_mos/2), self.num_mos-self.offset]
        )


    def _get_obs(self):
        """
            Reset the observation actually. 
        """
        self.observation = np.array([[0 for i in range(self.num_mos)],
                                     [0 for i in range(self.num_mos)]])
        return self.observation

    def _get_info(self):
        layout = []
        if self.placed_mos_pair != []:
            for mos_pair in self.placed_mos_pair:
                pair_info = []
                nmos_index = mos_pair[0]
                pmos_index = mos_pair[1]
                pair_info.append([
                    "0",    # nmos 
                    self.s_g_d[nmos_index][0] if self.observation[1][nmos_index] == 0 else self.s_g_d[nmos_index][2],
                    self.s_g_d[nmos_index][1],
                    self.s_g_d[nmos_index][2] if self.observation[1][nmos_index] == 0 else self.s_g_d[nmos_index][0],
                    self.mos_width[nmos_index],
                    self.mos_num_name[nmos_index]
                ])

                pair_info.append([
                    "1",
                    self.s_g_d[pmos_index][0] if self.observation[1][pmos_index] == 0 else self.s_g_d[pmos_index][2],
                    self.s_g_d[pmos_index][1],
                    self.s_g_d[pmos_index][2] if self.observation[1][pmos_index] == 0 else self.s_g_d[pmos_index][0],
                    self.mos_width[pmos_index],
                    self.mos_num_name[pmos_index]
                ])

                layout.append(pair_info)
        
        info = {
            "layout": layout
        }
            
        return info
    

    def _get_obs_step(self, action):
        if action[0] not in self.chosen_mos and action[1] not in self.chosen_mos:
            self.observation[0][action[0]] = 1
            self.observation[0][action[1]] = 1
        return self.observation

    def reset(self, seed=None, options=None):
        # 重置
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # self.stdcell_graph, _, _, _, _ ,_= get_stdcell_Graph(self.stdcell_name)
        observation = self._get_obs()
        info = self._get_info()

        self.chosen_mos = set()
        self.count = 0
        self.placed_mos_pair = []
        self.Q = 0
        self.info = {}
        self.nets = []

        return observation, info

    def is_action_repeated(self, action):
        repeat_num = 0
        if action[0] in self.chosen_mos:
            repeat_num += 1
        if action[1] in self.chosen_mos:
            repeat_num += 1
        return repeat_num
    
    def is_nmos_pmos_pair(self, action):
        if self.mos_num_name[action[0]] == self.mos_num_name[action[1]]:
            return 1
        else:
            return 0
    
    def flip_mos(self, mos_index):
        self.observation[1][mos_index] = 1 if self.observation[1][mos_index] == 0 else 0

    def get_reward(self, action):
        reward = 0
        
        # 重复选择问题
        if self.is_action_repeated(action) == 0:
            reward += 10
        else:
            reward -= 50

        # 计算布局宽度
        if self.placed_mos_pair != []:
            last_placed_mos_pair = self.placed_mos_pair[len(self.placed_mos_pair)-1]
            left_nmos_index = last_placed_mos_pair[0]
            left_pmos_index = last_placed_mos_pair[1]
            right_nmos_index = action[0]
            right_pmos_index = action[1]

            # The left side of this nmos may be source if it hasn't been flipped else drain. 
            left_1 = self.s_g_d[left_nmos_index][2] if self.observation[1][left_nmos_index] == 0 else self.s_g_d[left_nmos_index][0]
            right_1 = self.s_g_d[left_nmos_index][0] if self.observation[1][left_nmos_index] == 0 else self.s_g_d[left_nmos_index][2]

            left_2 = self.s_g_d[right_nmos_index][2] if self.observation[1][right_nmos_index] == 0 else self.s_g_d[right_nmos_index][0]
            right_2 = self.s_g_d[right_nmos_index][0] if self.observation[1][right_nmos_index] == 0 else self.s_g_d[right_nmos_index][2]

            left_3 = self.s_g_d[left_pmos_index][2] if self.observation[1][left_pmos_index] == 0 else self.s_g_d[left_pmos_index][0]
            right_3 = self.s_g_d[left_pmos_index][0] if self.observation[1][left_pmos_index] == 0 else self.s_g_d[left_pmos_index][2]

            left_4 = self.s_g_d[right_pmos_index][2] if self.observation[1][right_pmos_index] == 0 else self.s_g_d[right_pmos_index][0]
            right_4 = self.s_g_d[right_pmos_index][0] if self.observation[1][right_pmos_index] == 0 else self.s_g_d[right_pmos_index][2]

            # Best Condtion: left_2 = right_1 && left_4 = right_3 with no need to flip mos
            cond1 = (left_2 == right_1) and (left_4 == right_3)
            # left_2 = right_1 && (left_4 != right_3 && right_4 = right_3) with need to flip right pmos
            cond2 = (left_2 == right_1) and (left_4 != right_3 and right_4 == right_3)
            # (left_2 != right_1 && right_2 = right_1) && left_4 = right_3 with need to flip right nmos
            cond3 = (left_2 != right_1 and right_2 == right_1) and (left_4 == right_3)
            # (left_2 != right_1 && right_2 = right_1) && (left_4 != right_3 && right_4 = right_3) with need to flip both the right mos pair
            cond4 = (left_2 != right_1 and right_2 == right_1) and (left_4 != right_3 and right_4 == right_3)
            # Dummy PMOS and left_2 = right_1
            cond5 = left_4.split("_")[0] == "Dummy" and (left_2 == right_1) 
            # Dummy NMOS and left_4 = right_3
            cond6 = left_2.split("_")[0] == "Dummy" and (left_4 == right_3)
            # Dummy PMOS and (left_2 != right_1) && (right_2 = right_1) with need to flip right nmos
            cond7 = left_4.split("_")[0] == "Dummy" and ((left_2 != right_1 and right_2 == right_1))
            # Dummy NMOS and (left_4 != right_3) && (right_4 = right_3) with need to flip right pmos
            cond8 = left_2.split("_")[0] == "Dummy" and ((left_4 != right_3 and right_4 == right_3))

            if cond1 or cond5 or cond6:
                reward += 8
            elif cond2 or cond8:
                reward += 7
                self.flip_mos(right_pmos_index)
            elif cond3 or cond7:
                reward += 7
                self.flip_mos(right_nmos_index)
            elif cond4:
                reward += 6
                self.flip_mos(right_pmos_index)
                self.flip_mos(right_nmos_index)
            else:
                reward -= 3
        
        # 计算栅极未配对的情况
        if self.s_g_d[action[0]][1] == self.s_g_d[action[1]][1]:
            reward += 5
        else:
            reward -= 10
            self.not_share_gate_count += 1

        # DRC
        if len(self.info) >= 2:
            nmos_k_index = action[0]
            pmos_k_index = action[1]
            
            nmos_k_w = self.mos_width[nmos_k_index]
            pmos_k_w = self.mos_width[pmos_k_index] 
            nmos_j_w = self.info[-1][0][4]
            pmos_j_w = self.info[-1][1][4]
            nmos_i_w = self.info[-2][0][4]
            pmos_i_w = self.info[-2][1][4]

            nmos_k_left = self.s_g_d[nmos_k_index][0] if self.observation[1][nmos_k_index] == 0 else self.s_g_d[nmos_k_index][2]
            nmos_j_left = self.info[-1][0][1]
            nmos_j_right = self.info[-1][0][3]
            nmos_i_right = self.info[-2][0][3]

            pmos_k_left = self.s_g_d[pmos_k_index][0] if self.observation[1][pmos_k_index] == 0 else self.s_g_d[pmos_k_index][2]
            pmos_j_left = self.info[-1][1][1]
            pmos_j_right = self.info[-1][1][3]
            pmos_i_right = self.info[-2][1][3]

            if nmos_k_left == nmos_j_right and nmos_j_left == nmos_i_right:
                if nmos_j_w < nmos_k_w and nmos_j_w < nmos_i_w:
                    reward -= 100
            
            if pmos_k_left == pmos_j_right and pmos_j_left == pmos_i_right:
                if pmos_j_w < pmos_k_w and pmos_j_w < pmos_i_w:
                    reward -= 100
            
        
        # 布线复杂度
        # for j in range(0, 3):
        #     for i in range(len(self.nets)):
        #         nmos_net = self.s_g_d[action[0]][j]
        #         if nmos_net != "VDD" and nmos_net != "VSS":
        #             if nmos_net in self.nets[i]:
        #                 # print(f"nmos: {nmos_net}: {len(self.nets) - i}")
        #                 reward += (-(len(self.nets) - i) * 0.2) + 3 * 0.2 # 3是调节参数，即距离大于3的才会被惩罚
        #                 break
            
        #     for k in range(len(self.nets)):
        #         pmos_net = self.s_g_d[action[1]][j]
        #         if pmos_net != "VDD" and pmos_net != "VSS":
        #             if pmos_net in self.nets[k]:
        #                 # print(f"pmos: {pmos_net}: {len(self.nets) - i}")
        #                 reward += (-(len(self.nets) - k) * 0.2) + 3 * 0.2
        #                 break
        
        # nets_mos_pair = []
        # for i in range(0, 3):
        #     nmos_net = self.s_g_d[action[0]][i]
        #     if nmos_net != "VDD" and nmos_net != "VSS":
        #         nets_mos_pair.append(nmos_net)
        #     pmos_net = self.s_g_d[action[1]][i]
        #     if pmos_net != "VDD" and pmos_net != "VSS":
        #         nets_mos_pair.append(pmos_net)

        # self.nets.append(nets_mos_pair)


        return reward
    

    def step(self, action):
        # An episode is done if the mos pairs are all placed
        terminated = (self.num_of_episode == self.count)

        reward = 0
        observation = 0
        info = {}

        real_action = [0, 0]
        real_action[0] = action[0]
        real_action[1] = action[1] + self.offset

        if terminated:
            self.count = 0
            print(self.not_share_gate_count)
            print(self.Q)
            # print(f"The Num of not sharing gate mos pairs: {self.not_share_gate_count}")
            # print(f"Q Value: {self.Q}")
            self.not_share_gate_count = 0
        else:
            self.count += 1
            reward = self.get_reward(real_action)
            self.placed_mos_pair.append([real_action[0], real_action[1]])   
            self.Q += reward
            observation = self._get_obs_step(real_action)
            info = self._get_info()
            self.info = info["layout"]

            self.chosen_mos.add(real_action[0])
            self.chosen_mos.add(real_action[1])
        
        return observation, reward, terminated, False, info
    
