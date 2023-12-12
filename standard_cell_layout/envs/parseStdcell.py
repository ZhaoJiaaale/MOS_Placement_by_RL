import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import os

def get_channel_type(model):
    return 1 if model == "pch_mac" else 0

# 布局器类
class Layouter:
    def __init__(self, netlist_file, cell_name):
        self.netlist_file = netlist_file
        self.stdcell: StdCell = StdCell(cell_name)

    # 网表解析函数
    def parse_cdl(self):
        with open(self.netlist_file, 'r') as cdl:
            find_cell = False
            mos_num = 0
            while True:
                line = cdl.readline()
                if not line:
                    break
                words = line.split()
                if find_cell and words[0] == '.ENDS':
                    break

                if find_cell:
                    # 获取mos管宽度
                    mos_w = 0.00
                    for word in words[6:]:
                        if word[0] == 'w' or word[0] == 'W':
                            width_str = word.split('=')[1]
                            width = float(width_str[:-1])
                            unit = width_str[-1]
                            mos_w = int(width * 1000 if unit.upper() == 'U' else width)
                            self.stdcell.list_of_init_channel_width.append(mos_w)
                            break
                    # mos管信息
                    mos_name = words[0][1:]
                    mos_s = words[3]
                    mos_g = words[2]
                    mos_d = words[1]
                    mos_type = get_channel_type(words[5])
                    # 实例mos管
                    mos = Mos(mos_name, mos_num, 0, mos_type, mos_s, mos_g, mos_d, mos_w)
                    # 添加mos管
                    self.stdcell.add_mos(mos, mos_type)
                    mos_num += 1
                # 找到指定单元
                if len(words) > 1 and words[1] == self.stdcell.name:
                    find_cell = True

# 标准单元类
class StdCell:
    def __init__(self, name):
        self.name = name            # 标准单元名字
        self.num_of_nmos = 0        # nmos个数
        self.num_of_pmos = 0        # pmos个数
        self.num_of_pin = 0         # pin个数
        self.num_of_net = 0         # net个数
        self.num_of_gate = 0        # gate个数
        self.num_of_mos_pair = 0    # mos管对个数
        self.cnt_asymmetry = 0      # 不对称统计
        self.max_x = 0              # 最大横坐标
        self.list_of_nmos = []      # nmos列表
        self.list_of_pmos = []      # pmos列表
        self.list_of_pin = []       # pin列表
        self.head_mos_pair = None   # mos管对链表头
        self.dic_of_gate = {}       # gate集合
        self.dic_of_net = {}        # net字典
        self.list_of_init_channel_width = []    # 初始沟道长度列表

    # 添加mos管函数
    def add_mos(self, mos, type):
        if type == 1:
            self.list_of_pmos.append(mos)
            self.num_of_pmos += 1
        else:
            self.list_of_nmos.append(mos)
            self.num_of_nmos += 1

    # 产生初始管对
    def generate_initial_mos_pair(self):
        pass

    # 进行初始布局
    def generate_initial_layout(self):
        pass

    # 给mos管赋坐标
    def give_mos_x(self):
        return

    # 转为json格式
    def to_json(self):
        return {
            mos.name: mos.to_json() for mos in self.list_of_nmos + self.list_of_pmos
        }

# mos管类
class Mos:
    def __init__(self, name, num, x, type, source, gate, drain, width):
        self.name = name        # mos管名字
        self.num = num          # mos管编号
        self.x = x              # mos管横坐标
        self.type = type        # mos管类型(0-nmos,1-pmos)
        self.source = source    # mos管源极
        self.gate = gate        # mos管栅极
        self.drain = drain      # mos管漏极
        self.width = width      # mos管宽度（单位为nm）

    # 转为json格式
    def to_json(self):
        return {
            "x": str(self.x),
            "y": str(self.type),
            "source": self.source,
            "gate": self.gate,
            "drain": self.drain,
            "width": str(self.width)
        }


def get_stdcell_Graph(stdcell_name):
    layouter = Layouter("./Cells/cells.spi", stdcell_name)
    layouter.parse_cdl()
    nmos_list = layouter.stdcell.list_of_nmos
    pmos_list = layouter.stdcell.list_of_pmos

    new_nmos_list = []
    new_pmos_list = []
    # fold mos 
    
    nmos_num_count = 0
    for nmos in nmos_list:
        nmos.num = nmos_num_count
        nmos_num_count += 1
        if nmos.width > 220:
            print(f"Fold NMOS :{nmos.name}: {nmos.width} to")
            finger_width = int(nmos.width / 2) if nmos.width % 2 == 0 else int(nmos.width / 2) + 1
            nmos.width -= finger_width
            fold_mos = Mos(nmos.name + "_finger", nmos_num_count, nmos.x, nmos.type, nmos.source, nmos.gate, nmos.drain, finger_width)
            nmos_num_count += 1

            new_nmos_list.append(nmos)
            new_nmos_list.append(fold_mos)
            print(f"{nmos.name}: {nmos.width} and {fold_mos.name}: {fold_mos.width}")
        else:
            new_nmos_list.append(nmos)
    
    pmos_num_count = 0
    for pmos in pmos_list:
        pmos.num = pmos_num_count
        pmos_num_count += 1
        if pmos.width > 220:
            print(f"Fold PMOS :{pmos.name}: {pmos.width} to")
            finger_width = int(pmos.width / 2) if pmos.width % 2 == 0 else int(pmos.width / 2) + 1
            pmos.width -= finger_width
            fold_mos = Mos(pmos.name + "_finger", pmos_num_count, pmos.x, pmos.type, pmos.source, pmos.gate, pmos.drain, finger_width)
            pmos_num_count += 1
            
            new_pmos_list.append(pmos)
            new_pmos_list.append(fold_mos)
            print(f"{pmos.name}: {pmos.width} and {fold_mos.name}: {fold_mos.width}")
        else:
            new_pmos_list.append(pmos)
    
    if len(new_nmos_list) > len(new_pmos_list):
        # Add Dummy PMOS
        num_dummy_pmos = len(new_nmos_list) - len(new_pmos_list)

        for i in range(num_dummy_pmos):
            pmos_name = "Dummy_PMOS_" + str(i)
            dummy_pmos = Mos(pmos_name, pmos_num_count, 0, 1, "Dummy_Source", "Dummy_Gate", "Dummy_Drain", 100)
            pmos_num_count += 1
            new_pmos_list.append(dummy_pmos)

        print(f"Add {num_dummy_pmos} Dummy PMOS !")
    elif len(new_nmos_list) < len(new_pmos_list):
        # Add Dummy NMOS
        num_dummy_nmos = len(new_pmos_list) - len(new_nmos_list)

        for i in range(num_dummy_nmos):
            nmos_name = "Dummy_NMOS_" + str(i)
            dummy_nmos = Mos(nmos_name, nmos_num_count, 0, 0, "Dummy_Source_" + str(i), "Dummy_Gate" + str(i), "Dummy_Drain" + str(i), 100)
            nmos_num_count += 1
            new_nmos_list.append(dummy_nmos)
        print(f"Add {num_dummy_nmos} Dummy NMOS !")
    else:
        # Without Dummy MOS
        print("Without any Dummy MOS")

    for pmos in new_pmos_list:
        pmos.num += len(new_nmos_list)

    for nmos in new_nmos_list:
        print(nmos.name, nmos.num, nmos.source, nmos.gate, nmos.drain, nmos.width)
    for pmos in new_pmos_list:
        print(pmos.name, pmos.num, pmos.source, pmos.gate, pmos.drain, pmos.width)


    s_g_d = [[] for i in range(len(new_nmos_list)+len(new_pmos_list))]
    mos_width = [0 for i in range(len(new_nmos_list)+len(new_pmos_list))]    

    mos_num_name = {}
    for mos in new_nmos_list:
        mos_num_name[mos.num] = mos.name # + (" nmos" if mos.type == 0 else " pmos")
        s_g_d[mos.num] = [mos.source, mos.gate, mos.drain]
        mos_width[mos.num] = mos.width
        
    for mos in new_pmos_list:
        mos_num_name[mos.num] = mos.name # + (" nmos" if mos.type == 0 else " pmos")
        s_g_d[mos.num] = [mos.source, mos.gate, mos.drain]
        mos_width[mos.num] = mos.width

    return mos_num_name, new_nmos_list, new_pmos_list, s_g_d, mos_width
