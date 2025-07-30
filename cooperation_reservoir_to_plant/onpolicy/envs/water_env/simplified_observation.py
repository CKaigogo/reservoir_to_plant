"""
简化观测空间 - 减少观测维度，提高学习效率
"""

import numpy as np
from collections import deque
from typing import Dict, List


class SimplifiedObservationManager:
    """简化的观测管理器"""
    
    def __init__(self, num_reservoirs, num_plants, connections):
        self.num_reservoirs = num_reservoirs
        self.num_plants = num_plants
        self.connections = connections
        
        # 简化的观测
        self.obs_config = {
            'reservoir': {
                'self_state': 4,
                'system_state': 3,
                'time_info': 2,
                'history': 2
            },
            'plant': {
                'self_state': 3,
                'reservoir_state': 3,
                'system_state': 2,
                'time_info': 2,
                'history': 2
            }
        }
        
        # 计算简化后的观测维度
        self.reservoir_obs_dim = sum(self.obs_config['reservoir'].values())
        self.plant_obs_dim = sum(self.obs_config['plant'].values())
        self.max_obs_dim = max(self.reservoir_obs_dim, self.plant_obs_dim)
        
        # 历史状态缓存
        self.history_buffer = {
            'supply_satisfaction': deque(maxlen=5),
            'reservoir_levels': deque(maxlen=5),
        }
        
        # print(f" 简化观测空间：水库{self.reservoir_obs_dim}维，水厂{self.plant_obs_dim}维，最大{self.max_obs_dim}维")
        # print(f"   - 观测维度减少: 30维 → {self.max_obs_dim}维 (减少{30-self.max_obs_dim}维)")

    def get_simplified_observation(self, agent_id, env_state):
        """获取简化观测"""
        agent_type = agent_id.split('_')[0]
        agent_idx = int(agent_id.split('_')[1])

        if agent_type == 'reservoir':
            return self._get_simple_reservoir_observation(agent_idx, env_state)
        else:
            return self._get_simple_plant_observation(agent_idx, env_state)

    def _get_simple_reservoir_observation(self, res_id, env_state):
        """获取简化的水库观测"""
        obs_parts = []

        # 1.简化自身状态 (4维)
        level_ratio = env_state['reservoirs'][res_id] / env_state['max_reservoir'][res_id]
        available_ratio = max(0, env_state['reservoirs'][res_id] - env_state['dead_capacity'][res_id]) / \
                          env_state['max_reservoir'][res_id]
        
        # 简化的预警信号
        emergency_level = 1.0 if level_ratio < 0.2 else 0.5 if level_ratio < 0.4 else 0.0
        
        # 简化的节水信号
        conservation_signal = 1.0 if level_ratio < 0.3 else 0.0
        
        self_state = np.array([
            level_ratio,
            available_ratio,
            emergency_level,
            conservation_signal
        ])
        obs_parts.append(self_state)

        # 2. 简化系统状态
        # 全局供需平衡
        total_supply = np.sum(env_state.get('actual_supply', [0]))
        total_demand = np.sum(env_state.get('hourly_demand', [1]))
        supply_ratio = min(total_supply / (total_demand + 1e-8), 1.0)
        
        # 系统平均水位
        avg_reservoir_level = np.mean(env_state['reservoirs'] / env_state['max_reservoir'])
        
        # 系统压力指标
        low_level_count = np.sum((env_state['reservoirs'] / env_state['max_reservoir']) < 0.3)
        system_pressure = low_level_count / self.num_reservoirs
        
        system_state = np.array([
            supply_ratio,
            avg_reservoir_level,
            system_pressure
        ])
        obs_parts.append(system_state)

        # 3.简化时间信息
        current_hour = env_state.get('current_hour', 0)
        season_progress = env_state.get('season_progress', 0.0)
        
        time_state = np.array([
            np.sin(2 * np.pi * current_hour / 24),      # 小时周期
            np.sin(2 * np.pi * season_progress)         # 季节周期
        ])
        obs_parts.append(time_state)

        # 4. 历史信息
        current_satisfaction = self.history_buffer['supply_satisfaction'][-1] if self.history_buffer['supply_satisfaction'] else 0.5
        current_level = self.history_buffer['reservoir_levels'][-1] if self.history_buffer['reservoir_levels'] else 0.5
        
        history_state = np.array([current_satisfaction, current_level])
        obs_parts.append(history_state)

        full_obs = np.concatenate(obs_parts)
        padded_obs = np.zeros(self.max_obs_dim, dtype=np.float32)
        padded_obs[:len(full_obs)] = full_obs

        return padded_obs

    def _get_simple_plant_observation(self, plant_id, env_state):
        """获取简化的水厂观测"""
        obs_parts = []

        # 1.简化自身状态
        demand = env_state['hourly_demand'][plant_id]
        supply = env_state['actual_supply'][plant_id]
        inventory = env_state['plant_inventory'][plant_id]
        storage_capacity = env_state['plant_storage_capacity'][plant_id]

        self_state = np.array([
            supply / (demand + 1e-8),           # 供需比
            inventory / storage_capacity,        # 库存比例
            1.0 if supply >= demand else 0.0   # 满足标志
        ])
        obs_parts.append(self_state)

        # 2.简化水库状态
        connected_reservoirs = np.where(self.connections[:, plant_id])[0]
        if len(connected_reservoirs) > 0:
            reservoir_levels = [env_state['reservoirs'][r] / env_state['max_reservoir'][r] for r in connected_reservoirs]
            
            reservoir_state = np.array([
                np.mean(reservoir_levels),  # 平均水位
                np.min(reservoir_levels),   # 最低水位
                len(connected_reservoirs) / self.num_reservoirs
            ])
        else:
            reservoir_state = np.zeros(3)
        obs_parts.append(reservoir_state)

        # 3.简化系统状态
        # 全局供需平衡
        total_supply = np.sum(env_state['actual_supply'])
        total_demand = np.sum(env_state['hourly_demand'])
        
        # 系统平均水位
        avg_reservoir_level = np.mean(env_state['reservoirs'] / env_state['max_reservoir'])
        
        system_state = np.array([
            total_supply / (total_demand + 1e-8),
            avg_reservoir_level
        ])
        obs_parts.append(system_state)

        # 4. 时间信息
        current_hour = env_state.get('current_hour', 0)
        season_progress = env_state.get('season_progress', 0.0)
        
        time_state = np.array([
            np.sin(2 * np.pi * current_hour / 24),
            np.sin(2 * np.pi * season_progress)
        ])
        obs_parts.append(time_state)

        # 5.简化历史信息
        current_satisfaction = self.history_buffer['supply_satisfaction'][-1] if self.history_buffer['supply_satisfaction'] else 0.5
        current_level = self.history_buffer['reservoir_levels'][-1] if self.history_buffer['reservoir_levels'] else 0.5
        
        history_state = np.array([current_satisfaction, current_level])
        obs_parts.append(history_state)

        # 组合并填充到统一维度
        full_obs = np.concatenate(obs_parts)
        padded_obs = np.zeros(self.max_obs_dim, dtype=np.float32)
        padded_obs[:len(full_obs)] = full_obs

        return padded_obs

    def update_history(self, env_state):
        """更新历史缓存"""
        # 计算全局指标
        total_supply = np.sum(env_state.get('actual_supply', [0]))
        total_demand = np.sum(env_state.get('hourly_demand', [1]))
        satisfaction = min(total_supply / (total_demand + 1e-8), 1.0)
        avg_level = np.mean(env_state['reservoirs'] / env_state['max_reservoir'])

        # 更新历史
        self.history_buffer['supply_satisfaction'].append(satisfaction)
        self.history_buffer['reservoir_levels'].append(avg_level)

    def reset_history(self):
        """重置历史缓存"""
        self.history_buffer['supply_satisfaction'].clear()
        self.history_buffer['reservoir_levels'].clear()

    def get_observation_summary(self):
        """获取观测空间摘要"""
        return {
            'reservoir_obs_dim': self.reservoir_obs_dim,
            'plant_obs_dim': self.plant_obs_dim,
            'max_obs_dim': self.max_obs_dim,
            'dimension_reduction': f"30 → {self.max_obs_dim}",
            'components': {
                'reservoir': self.obs_config['reservoir'],
                'plant': self.obs_config['plant']
            }
        } 