import numpy as np
from gym import spaces
from typing import Dict, List, Any, Tuple


class SimplifiedActionProcessor:
    """简化的动作处理器 - 将连续动作空间简化为离散选择"""

    def __init__(self, num_reservoirs, num_plants):
        self.num_reservoirs = num_reservoirs
        self.num_plants = num_plants

        # 预定义的动作映射
        self.reservoir_release_levels = [0.05, 0.1, 0.2, 0.3, 0.5]  # 5个释放等级
        self.plant_demand_levels = [0.8, 1.0, 1.2]  # 3个需求等级

    def get_simplified_action_spaces(self):
        """获取简化的动作空间定义"""
        action_spaces = {}

        # 水库动作简化
        for i in range(self.num_reservoirs):
            agent_id = f"reservoir_{i}"
            action_spaces[agent_id] = spaces.Dict({
                'release_level': spaces.Discrete(5),
                'emergency_mode': spaces.Discrete(2)  # 是否紧急 (0/1)
            })

        # 水厂动作简化
        for i in range(self.num_plants):
            agent_id = f"plant_{i}"
            action_spaces[agent_id] = spaces.Dict({
                'demand_level': spaces.Discrete(3),
                'priority': spaces.Discrete(2)
            })

        return action_spaces

    def convert_simplified_to_original_actions(self, simplified_actions):
        """将简化动作转换为原始环境动作格式"""
        converted_actions = {}

        for agent_id, action in simplified_actions.items():
            if 'reservoir' in agent_id:
                # 处理水库动作
                if isinstance(action, dict):
                    release_level_idx = action.get('release_level', 0)
                    emergency_mode = action.get('emergency_mode', 0)
                    allocation_preference = action.get('allocation_preference', 0.5)
                else:
                    # 数组格式输入
                    if len(action) >= 3:
                        release_level_idx = int(action[0]) if action[0] < 5 else int(action[0] % 5)
                        emergency_mode = int(action[1]) if action[1] < 2 else int(action[1] % 2)
                        allocation_preference = float(action[2])
                    else:
                        release_level_idx = int(action[0]) if len(action) > 0 else 0
                        emergency_mode = int(action[1]) if len(action) > 1 else 0
                        allocation_preference = 0.5

                # 转换为原始动作格式
                release_ratio = self.reservoir_release_levels[release_level_idx]
                
                # 根据偏好生成分配权重
                allocation_weights = self._generate_allocation_weights(allocation_preference)

                converted_actions[agent_id] = {
                    'total_release_ratio': np.array([release_ratio]),
                    'allocation_weights': allocation_weights,
                    'emergency_release': emergency_mode
                }

            else:  # plant
                # 处理水厂动作
                if isinstance(action, dict):
                    demand_level_idx = action.get('demand_level', 1)
                    priority = action.get('priority', 0)
                    storage_preference = action.get('storage_preference', 0.5)
                else:
                    if len(action) >= 3:
                        demand_level_idx = int(action[0]) if action[0] < 3 else int(action[0] % 3)
                        priority = int(action[1]) if action[1] < 2 else int(action[1] % 2)
                        storage_preference = float(action[2])
                    else:
                        demand_level_idx = int(action[0]) if len(action) > 0 else 1
                        priority = int(action[1]) if len(action) > 1 else 0
                        storage_preference = 0.5

                # 转换为原始动作格式
                demand_factor = self.plant_demand_levels[demand_level_idx]

                converted_actions[agent_id] = {
                    'demand_adjustment': np.array([demand_factor]),
                    'priority_level': priority,
                    'storage_strategy': np.array([storage_preference])
                }

        return converted_actions

    def _generate_allocation_weights(self, preference):
        """根据偏好生成分配权重"""
        if self.num_plants == 1:
            return np.array([1.0])
        
        # 基于偏好生成权重分布
        if preference < 0.33:
            # 偏好第一个水厂
            weights = np.zeros(self.num_plants)
            weights[0] = 0.7
            weights[1:] = 0.3 / (self.num_plants - 1)
        elif preference < 0.67:
            # 均匀分配
            weights = np.ones(self.num_plants) / self.num_plants
        else:
            # 偏好最后一个水厂
            weights = np.zeros(self.num_plants)
            weights[-1] = 0.7
            weights[:-1] = 0.3 / (self.num_plants - 1)
        
        return weights

    def get_action_space_for_mappo(self):
        """获取统一的动作空间"""
        reservoir_action_dim = 3  # release_level + emergency_mode + allocation_preference
        plant_action_dim = 3      # demand_level + priority + storage_preference

        # 使用最大维度创建统一空间
        max_action_dim = max(reservoir_action_dim, plant_action_dim)

        action_spaces = []
        for i in range(self.num_reservoirs + self.num_plants):
            # 使用 Box 空间，MAPPO 需要连续空间
            action_spaces.append(
                spaces.Box(low=0.0, high=1.0, shape=(max_action_dim,), dtype=np.float32)
            )

        return action_spaces

    def convert_mappo_actions_to_simplified(self, mappo_actions, agent_ids):
        """将 MAPPO 输出的连续动作转换为简化的离散动作"""
        simplified_actions = {}

        for i, agent_id in enumerate(agent_ids):
            raw_action = mappo_actions[i]

            if 'reservoir' in agent_id:
                # 水库动作：将连续值映射到离散选择
                release_level = int(raw_action[0] * 5) % 5
                emergency_mode = 1 if raw_action[1] > 0.5 else 0
                
                # 分配偏好
                allocation_preference = raw_action[2] if len(raw_action) > 2 else 0.5

                simplified_actions[agent_id] = {
                    'release_level': release_level,
                    'emergency_mode': emergency_mode,
                    'allocation_preference': allocation_preference
                }

            else:  # plant
                # 水厂动作：将连续值映射到离散选择
                demand_level = int(raw_action[0] * 3) % 3
                priority = 1 if raw_action[1] > 0.5 else 0

                storage_preference = raw_action[2] if len(raw_action) > 2 else 0.5

                simplified_actions[agent_id] = {
                    'demand_level': demand_level,
                    'priority': priority,
                    'storage_preference': storage_preference
                }

        return simplified_actions