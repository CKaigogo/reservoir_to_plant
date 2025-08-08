"""
多智能体水资源管理环境
"""

import gym
import numpy as np
import pandas as pd
import pygame

from pettingzoo import ParallelEnv
from gym import spaces
import math
from collections import deque, defaultdict
import os
import torch
import copy
from typing import Dict, Tuple, List, Any, Optional


from .reward_system import StabilizedMultiAgentRewardSystem

fixed_connections = np.array([
    [True, False, False, False, False, False],   # reservoir_0 -> plant_0
    [False, False, True, False, False, False],   # reservoir_1 -> plant_2
    [False, False, True, False, False, False],   # reservoir_2 -> plant_2
    [False, True, False, False, False, False],   # reservoir_3 -> plant_1
    [False, True, False, False, False, False],   # reservoir_4 -> plant_1
    [False, True, False, False, False, False],   # reservoir_5 -> plant_1
    [False, False, False, True, False, False],
    [False, False, False, False, True, False],
    [True, False, False, False, False, False],   # reservoir_8 -> plant_0
    [False, False, False, False, False, True],
    [True, False, False, False, False, False],   # reservoir_10 -> plant_0
    [True, False, False, False, False, False],   # reservoir_11 -> plant_0
    [True, False, False, False, False, False],   # reservoir_12 -> plant_0
    [True, False, False, False, False, False],   # reservoir_13 -> plant_0
    [False, True, False, False, False, False],   # reservoir_14 -> plant_1
])


class ActionProcessor:
    """动作处理器 """

    def __init__(self, num_reservoirs, num_plants, connections):
        self.num_reservoirs = num_reservoirs
        self.num_plants = num_plants
        self.connections = connections

        # 预计算每个水库的连接信息
        self.reservoir_connections = {}
        for i in range(num_reservoirs):
            connected_plants = np.where(connections[i, :])[0]
            self.reservoir_connections[i] = {
                'plants': connected_plants.tolist(),
                'count': len(connected_plants)
            }

    def rehydrate_actions_fixed(self, raw_actions):
        """动作重构方法"""
        hydrated_actions = {}

        for agent_id, raw_action in raw_actions.items():
            try:
                if isinstance(raw_action, dict):
                    hydrated_actions[agent_id] = self._standardize_dict_action(agent_id, raw_action)
                    continue

                action_dict = self._parse_flat_action(agent_id, raw_action)
                hydrated_actions[agent_id] = action_dict

            except Exception as e:
                print(f"警告：动作解析失败 {agent_id}: {e}")
                hydrated_actions[agent_id] = self._get_safe_default_action(agent_id)

        return hydrated_actions

    def _standardize_dict_action(self, agent_id, action_dict):
        """标准化字典格式的动作"""
        standardized = {}
        agent_type = agent_id.split("_")[0]

        if agent_type == "reservoir":
            standardized['total_release_ratio'] = self._ensure_array(
                action_dict.get('total_release_ratio', [0.1]), 1
            )

            # 所有水库都使用相同的分配权重维度（等于水厂总数）
            raw_weights = action_dict.get('allocation_weights', np.ones(self.num_plants))
            standardized['allocation_weights'] = self._ensure_array(raw_weights, self.num_plants)

            emergency_val = action_dict.get('emergency_release', 0)
            if isinstance(emergency_val, (np.ndarray, list)):
                emergency_val = emergency_val[0] if len(emergency_val) > 0 else 0
            standardized['emergency_release'] = int(np.clip(emergency_val, 0, 1))

        elif agent_type == "plant":
            standardized['demand_adjustment'] = self._ensure_array(
                action_dict.get('demand_adjustment', [1.0]), 1
            )

            priority_val = action_dict.get('priority_level', 1)
            if isinstance(priority_val, (np.ndarray, list)):
                priority_val = priority_val[0] if len(priority_val) > 0 else 1
            standardized['priority_level'] = int(np.clip(priority_val, 0, 2))

            standardized['storage_strategy'] = self._ensure_array(
                action_dict.get('storage_strategy', [0.5]), 1
            )

        return standardized

    def _parse_flat_action(self, agent_id, raw_action):
        """解析动作数组"""
        if not isinstance(raw_action, np.ndarray):
            raw_action = np.array([raw_action]) if np.isscalar(raw_action) else np.array(raw_action)

        action_dict = {}
        agent_type, agent_idx_str = agent_id.split("_")
        current_idx = 0

        if agent_type == "reservoir":
            # 解析总释放比例
            if current_idx < len(raw_action):
                action_dict['total_release_ratio'] = np.array([
                    np.clip(float(raw_action[current_idx]), 0.0, 1.0)
                ])
                current_idx += 1
            else:
                action_dict['total_release_ratio'] = np.array([0.1])

            # 解析分配权重 - 使用固定维度
            if current_idx + self.num_plants <= len(raw_action):
                weights = raw_action[current_idx:current_idx + self.num_plants]
                weights = np.clip(weights, 0.0, 1.0)
                # 归一化权重
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = np.ones(self.num_plants) / self.num_plants
                action_dict['allocation_weights'] = weights
                current_idx += self.num_plants
            else:
                action_dict['allocation_weights'] = np.ones(self.num_plants) / self.num_plants

            # 解析紧急释放
            if current_idx < len(raw_action):
                emergency_val = float(raw_action[current_idx])
                action_dict['emergency_release'] = int(np.clip(np.round(emergency_val), 0, 1))
            else:
                action_dict['emergency_release'] = 0

        elif agent_type == "plant":
            # 解析需求调整
            if current_idx < len(raw_action):
                action_dict['demand_adjustment'] = np.array([
                    np.clip(float(raw_action[current_idx]), 0.5, 1.5)
                ])
                current_idx += 1
            else:
                action_dict['demand_adjustment'] = np.array([1.0])

            # 解析优先级
            if current_idx < len(raw_action):
                priority_val = float(raw_action[current_idx])
                action_dict['priority_level'] = int(np.clip(np.round(priority_val), 0, 2))
                current_idx += 1
            else:
                action_dict['priority_level'] = 1

            # 解析存储策略
            if current_idx < len(raw_action):
                action_dict['storage_strategy'] = np.array([
                    np.clip(float(raw_action[current_idx]), 0.0, 1.0)
                ])
            else:
                action_dict['storage_strategy'] = np.array([0.5])

        return action_dict

    def _ensure_array(self, value, expected_size):
        """确保值是指定大小的numpy数组"""
        if isinstance(value, (list, np.ndarray)):
            arr = np.array(value, dtype=np.float32)
            if len(arr) == expected_size:
                return arr
            elif len(arr) > expected_size:
                return arr[:expected_size]
            else:
                padded = np.zeros(expected_size, dtype=np.float32)
                padded[:len(arr)] = arr
                return padded
        else:
            return np.full(expected_size, float(value), dtype=np.float32)

    def _get_safe_default_action(self, agent_id):
        """获取安全的默认动作"""
        agent_type = agent_id.split("_")[0]

        if agent_type == "reservoir":
            return {
                'total_release_ratio': np.array([0.1]),
                'allocation_weights': np.ones(self.num_plants) / self.num_plants,
                'emergency_release': 0
            }
        else:
            return {
                'demand_adjustment': np.array([1.0]),
                'priority_level': 1,
                'storage_strategy': np.array([0.5])
            }


class ContinuousStateManager:
    """连续状态管理器"""

    def __init__(self, num_reservoirs, num_plants):
        self.num_reservoirs = num_reservoirs
        self.num_plants = num_plants
        self.max_overnight_change = 0.05
        self.max_seasonal_drift = 0.02
        self.min_valid_ratio = 0.01
        self.max_valid_ratio = 1.0

    def handle_continuous_reset(self, current_state, cumulative_days):
        """简化的连续重置逻辑"""
        new_state = copy.deepcopy(current_state)

        # 1. 水库状态平滑过渡
        new_state['reservoirs'] = self._smooth_reservoir_transition(
            current_state['reservoirs'],
            current_state['max_reservoir'],
            cumulative_days
        )

        # 2. 水厂需求渐进调整
        new_state['plants_demand'] = self._gradual_demand_adjustment(
            current_state['plants_demand'],
            current_state['max_plant'],
            cumulative_days
        )

        # 3. 水厂库存自然衰减
        new_state['plant_inventory'] = self._natural_inventory_decay(
            current_state['plant_inventory'],
            current_state['plant_storage_capacity']
        )

        # 4. 状态验证和修正
        new_state = self._validate_and_correct_state(new_state)

        return new_state

    def _smooth_reservoir_transition(self, reservoirs, max_reservoir, cumulative_days):
        """平滑的水库状态过渡"""
        new_reservoirs = reservoirs.copy()

        # 过夜自然变化
        overnight_factor = np.random.normal(1.0, 0.01, self.num_reservoirs)
        overnight_factor = np.clip(overnight_factor,
                                   1.0 - self.max_overnight_change,
                                   1.0 + self.max_overnight_change)
        new_reservoirs = new_reservoirs * overnight_factor

        # 季节性偏移
        season_progress = (cumulative_days / 365.0) % 1.0
        seasonal_factor = 1.0 + self.max_seasonal_drift * np.sin(2 * np.pi * season_progress)
        new_reservoirs = new_reservoirs * seasonal_factor

        # 确保在合理范围内
        min_levels = max_reservoir * 0.05
        max_levels = max_reservoir * 0.95
        new_reservoirs = np.clip(new_reservoirs, min_levels, max_levels)

        return new_reservoirs

    def _gradual_demand_adjustment(self, plants_demand, max_plant, cumulative_days):
        """渐进的需求调整"""
        new_demand = plants_demand.copy()

        # 基于星期模式的需求变化
        day_of_week = cumulative_days % 7
        weekly_factors = {0: 1.0, 1: 1.05, 2: 1.1, 3: 1.08, 4: 1.02, 5: 0.8, 6: 0.7}
        weekly_factor = weekly_factors.get(day_of_week, 1.0)

        # 季节性需求变化
        season_progress = (cumulative_days / 365.0) % 1.0
        seasonal_factor = 0.8 + 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * season_progress))

        # 随机波动
        random_factor = np.random.normal(1.0, 0.02, self.num_plants)

        # 综合调整
        adjustment = weekly_factor * seasonal_factor * random_factor
        new_demand = new_demand * adjustment

        # 限制在合理范围
        new_demand = np.clip(new_demand, 0.3 * max_plant, 1.5 * max_plant)

        return new_demand

    def _natural_inventory_decay(self, plant_inventory, plant_storage_capacity):
        """自然的库存衰减"""
        consumption_rate = np.random.uniform(0.02, 0.08, self.num_plants)
        new_inventory = plant_inventory * (1.0 - consumption_rate)
        min_inventory = plant_storage_capacity * 0.1
        new_inventory = np.maximum(new_inventory, min_inventory)
        return new_inventory

    def _validate_and_correct_state(self, state):
        """验证和修正状态"""
        corrected_state = copy.deepcopy(state)

        # 检查水库状态
        reservoir_ratios = corrected_state['reservoirs'] / corrected_state['max_reservoir']
        invalid_reservoirs = (reservoir_ratios < self.min_valid_ratio) | (reservoir_ratios > self.max_valid_ratio)

        if np.any(invalid_reservoirs):
            print(f"警告：检测到无效水库状态，进行修正")
            corrected_state['reservoirs'][invalid_reservoirs] = (
                corrected_state['max_reservoir'][invalid_reservoirs] * 0.5
            )

        return corrected_state


class StructuredObservationManager:
    """结构化观测管理器"""

    def __init__(self, num_reservoirs, num_plants, connections):
        self.num_reservoirs = num_reservoirs
        self.num_plants = num_plants
        self.connections = connections

        # 定义固定的观测维度
        self.obs_config = {
            'reservoir': {
                'self_state': 6,
                'connected_plants': 2,
                'global_info': 8,
                'time_info': 6,
                'neighbors': 4,
                'history': 4
            },
            'plant': {
                'self_state': 5,
                'connected_reservoirs': 6,
                'global_info': 8,
                'time_info': 6,
                'competitors': 0,         # 无竞争者
                'system_state': 2,
                'history': 4
            }
        }

        # 计算总维度
        self.reservoir_obs_dim = sum(self.obs_config['reservoir'].values())
        self.plant_obs_dim = sum(self.obs_config['plant'].values())
        self.max_obs_dim = max(self.reservoir_obs_dim, self.plant_obs_dim)

        # 历史状态缓存
        self.history_buffer = {
            'supply_satisfaction': deque(maxlen=10),
            'reservoir_levels': deque(maxlen=10),
            'demand_levels': deque(maxlen=10)
        }

        print(f"结构化观测空间：水库{self.reservoir_obs_dim}维，水厂{self.plant_obs_dim}维，最大{self.max_obs_dim}维")

    def get_structured_observation(self, agent_id, env_state):
        """获取结构化观测"""
        agent_type = agent_id.split('_')[0]
        agent_idx = int(agent_id.split('_')[1])

        if agent_type == 'reservoir':
            return self._get_reservoir_observation(agent_idx, env_state)
        else:
            return self._get_plant_observation(agent_idx, env_state)

    def _get_reservoir_observation(self, res_id, env_state):
        """获取水库智能体的结构化观测"""
        obs_parts = []

        # 1. 自身状态 (6维)
        level_ratio = env_state['reservoirs'][res_id] / env_state['max_reservoir'][res_id]
        available_ratio = max(0, env_state['reservoirs'][res_id] - env_state['dead_capacity'][res_id]) / \
                          env_state['max_reservoir'][res_id]

        # 预警信号
        emergency_level = 0.0
        if level_ratio < 0.20:  # 死水位
            emergency_level = 1.0  # 最高预警
        elif level_ratio < 0.30:  # 紧急水位
            emergency_level = 0.8  # 高预警
        elif level_ratio < 0.40:  # 警戒水位
            emergency_level = 0.5  # 中预警
        elif level_ratio < 0.50:  # 注意水位
            emergency_level = 0.2  # 低预警

        # 水位下降趋势预警
        trend_warning = 0.0
        if len(self.history_buffer['reservoir_levels']) >= 5:
            recent_levels = list(self.history_buffer['reservoir_levels'])[-5:]
            if len(recent_levels) >= 2:
                trend = (recent_levels[-1] - recent_levels[0]) / len(recent_levels)
                if trend < -0.02:  # 快速下降
                    trend_warning = 1.0
                elif trend < -0.01:  # 缓慢下降
                    trend_warning = 0.5

        # 系统性缺水预警
        avg_system_level = np.mean(env_state['reservoirs'] / env_state['max_reservoir'])
        system_scarcity = 1.0 if avg_system_level < 0.3 else 0.5 if avg_system_level < 0.5 else 0.0

        self_state = np.array([
            level_ratio,
            available_ratio,
            emergency_level,  # 紧急程度
            trend_warning,  # 下降趋势预警
            system_scarcity,  # 系统缺水预警
            1.0 if level_ratio < 0.3 else 0.0  # 节水模式信号
        ])
        obs_parts.append(self_state)

        # 2.优化连接水厂状态 (2维) - 针对单个水厂简化
        connected_plants = np.where(self.connections[res_id, :])[0]
        if len(connected_plants) > 0:
            plant_ratios = [env_state['actual_supply'][p] / (env_state['hourly_demand'][p] + 1e-8) for p in
                            connected_plants]
            connected_state = np.array([
                np.mean(plant_ratios),  # 平均满足率
                len(connected_plants) / self.num_plants  # 连接比例
            ])
        else:
            connected_state = np.zeros(2)
        obs_parts.append(connected_state)

        # 3-6
        obs_parts.extend([
            self._get_global_state(env_state),
            self._get_time_state(env_state),
            self._get_neighbor_state(res_id, env_state),
            self._get_history_state()
        ])

        # 组合并填充
        full_obs = np.concatenate(obs_parts)
        padded_obs = np.zeros(self.max_obs_dim, dtype=np.float32)
        padded_obs[:len(full_obs)] = full_obs

        return padded_obs

    def _get_plant_observation(self, plant_id, env_state):
        """获取水厂智能体的结构化观测"""
        obs_parts = []

        # 1. 自身状态 (5维)
        demand = env_state['hourly_demand'][plant_id]
        supply = env_state['actual_supply'][plant_id]
        inventory = env_state['plant_inventory'][plant_id]
        storage_capacity = env_state['plant_storage_capacity'][plant_id]

        self_state = np.array([
            supply / (demand + 1e-8),
            inventory / storage_capacity,
            demand / env_state['max_plant'][plant_id],
            1.0 if supply >= demand else 0.0,
            1.0 if inventory < 0.2 * storage_capacity else 0.0
        ])
        obs_parts.append(self_state)

        # 2. 连接水库状态 (5维)
        connected_reservoirs = np.where(self.connections[:, plant_id])[0]
        if len(connected_reservoirs) > 0:
            reservoir_levels = [env_state['reservoirs'][r] / env_state['max_reservoir'][r] for r in
                                connected_reservoirs]
            available_water = [max(0, env_state['reservoirs'][r] - env_state['dead_capacity'][r]) for r in
                               connected_reservoirs]

            reservoir_state = np.array([
                np.mean(reservoir_levels),
                np.min(reservoir_levels),
                sum(available_water) / (sum([env_state['max_reservoir'][r] for r in connected_reservoirs]) + 1e-8),
                len(connected_reservoirs) / self.num_reservoirs,
                np.std(reservoir_levels) if len(reservoir_levels) > 1 else 0.0
            ])
        else:
            reservoir_state = np.zeros(5)
        obs_parts.append(reservoir_state)

        # 3-6. 其他状态组件
        obs_parts.extend([
            self._get_global_state(env_state),
            self._get_time_state(env_state),
            self._get_system_state(plant_id, env_state),
            self._get_history_state()
        ])

        # 组合并填充
        full_obs = np.concatenate(obs_parts)
        padded_obs = np.zeros(self.max_obs_dim, dtype=np.float32)
        padded_obs[:len(full_obs)] = full_obs

        return padded_obs

    def _get_system_state(self, plant_id, env_state):
        """获取系统状态信息"""
        # 系统整体供需平衡
        total_supply = np.sum(env_state['actual_supply'])
        total_demand = np.sum(env_state['hourly_demand'])

        # 系统压力指标
        avg_reservoir_level = np.mean(env_state['reservoirs'] / env_state['max_reservoir'])

        return np.array([
            total_supply / (total_demand + 1e-8),  # 系统供需比
            avg_reservoir_level
        ])

    def _get_global_state(self, env_state):
        """获取全局状态信息"""
        total_supply = np.sum(env_state['actual_supply'])
        total_demand = np.sum(env_state['hourly_demand'])
        total_reservoir = np.sum(env_state['reservoirs'])
        total_capacity = np.sum(env_state['max_reservoir'])

        reservoir_levels = env_state['reservoirs'] / env_state['max_reservoir']

        return np.array([
            total_supply / (total_demand + 1e-8),
            total_reservoir / total_capacity,
            np.mean(reservoir_levels),
            np.std(reservoir_levels),
            np.sum(reservoir_levels > 0.8) / self.num_reservoirs,
            np.sum(reservoir_levels < 0.2) / self.num_reservoirs,
            np.sum(env_state['actual_supply'] >= env_state['hourly_demand']) / self.num_plants,
            min(total_supply / (total_demand + 1e-8), 1.0)
        ])

    def _get_time_state(self, env_state):
        """获取时间状态信息"""
        current_hour = env_state.get('current_hour', 0)
        season_progress = env_state.get('season_progress', 0.0)

        future_season_1 = (season_progress + 0.1) % 1.0  # 未来3天
        future_season_2 = (season_progress + 0.2) % 1.0  # 未来6天

        return np.array([
            np.sin(2 * np.pi * current_hour / 24),
            np.cos(2 * np.pi * current_hour / 24),
            np.sin(2 * np.pi * season_progress),
            np.cos(2 * np.pi * season_progress),
            np.sin(2 * np.pi * future_season_1),
            np.cos(2 * np.pi * future_season_2)
        ])

    def _get_neighbor_state(self, res_id, env_state):
        """获取邻居水库状态 - 适应性版本"""
        if self.num_reservoirs <= 4:
            # 小规模系统：包含所有其他水库
            neighbors = [i for i in range(self.num_reservoirs) if i != res_id]
        else:
            # 大规模系统：使用邻域策略
            neighbors = []
            for i in range(max(0, res_id - 2), min(self.num_reservoirs, res_id + 3)):
                if i != res_id:
                    neighbors.append(i)

        # 计算邻居状态
        if neighbors:
            neighbor_levels = [env_state['reservoirs'][n] / env_state['max_reservoir'][n] for n in neighbors]

            # 适应性统计计算
            if len(neighbor_levels) == 1:
                # 只有1个邻居的情况
                return np.array([
                    neighbor_levels[0],
                    neighbor_levels[0],
                    neighbor_levels[0],
                    0.0  # 标准差为0
                ])
            elif len(neighbor_levels) == 2:
                # 有2个邻居的情况
                return np.array([
                    np.mean(neighbor_levels),
                    np.max(neighbor_levels),
                    np.min(neighbor_levels),
                    abs(neighbor_levels[0] - neighbor_levels[1]) / 2
                ])
            else:
                # 有3个或更多邻居的情况
                return np.array([
                    np.mean(neighbor_levels),
                    np.max(neighbor_levels),
                    np.min(neighbor_levels),
                    np.std(neighbor_levels)
                ])
        else:
            return np.zeros(4)

    def _get_competitor_state(self, plant_id, env_state):
        """获取竞争者状态"""
        if self.num_plants == 1:
            # 单个水厂情况：返回全局竞争信息
            return np.array([
                1.0,  # 无竞争标志
                env_state['actual_supply'][plant_id] / (env_state['hourly_demand'][plant_id] + 1e-8),
                0.0   # 保留维度
            ])
        else:
            # 多个水厂情况：返回实际竞争信息
            other_plants = [i for i in range(self.num_plants) if i != plant_id]
            if other_plants:
                other_demands = [env_state['hourly_demand'][p] for p in other_plants]
                other_supplies = [env_state['actual_supply'][p] for p in other_plants]

                return np.array([
                    sum(other_demands) / (sum(env_state['max_plant']) - env_state['max_plant'][plant_id] + 1e-8),
                    np.mean([s / (d + 1e-8) for s, d in zip(other_supplies, other_demands)]),
                    sum(other_supplies) / (sum(other_demands) + 1e-8)
                ])
            else:
                return np.zeros(3)

    def _get_history_state(self):
        """获取历史状态信息 (4维)"""
        if len(self.history_buffer['supply_satisfaction']) >= 3:
            recent_satisfaction = list(self.history_buffer['supply_satisfaction'])[-3:]
            satisfaction_trend = (recent_satisfaction[-1] - recent_satisfaction[0]) / 3
        else:
            satisfaction_trend = 0.0

        current_satisfaction = self.history_buffer['supply_satisfaction'][-1] if self.history_buffer['supply_satisfaction'] else 0.5
        current_level = self.history_buffer['reservoir_levels'][-1] if self.history_buffer['reservoir_levels'] else 0.5

        return np.array([current_satisfaction, satisfaction_trend, current_level, 0.0])

    def update_history(self, env_state):
        """更新历史缓存"""
        total_supply = np.sum(env_state['actual_supply'])
        total_demand = np.sum(env_state['hourly_demand'])
        satisfaction = min(total_supply / (total_demand + 1e-8), 1.0)
        avg_level = np.mean(env_state['reservoirs'] / env_state['max_reservoir'])

        self.history_buffer['supply_satisfaction'].append(satisfaction)
        self.history_buffer['reservoir_levels'].append(avg_level)

    def reset_history(self):
        """重置历史缓存"""
        self.history_buffer['supply_satisfaction'].clear()
        self.history_buffer['reservoir_levels'].clear()
        self.history_buffer['demand_levels'].clear()


class WaterManagementEnv(ParallelEnv):
    """优化的多智能体水资源管理环境"""

    def __init__(self, num_reservoirs=12, num_plants=3, max_episode_steps=96,
                 continuous_management=False, progressive_training_manager=None,
                 use_fixed_connections=True, enable_optimizations=True):
        """初始化水资源管理环境"""
        super().__init__()

        # ==================== 基础环境参数 ====================
        self.num_reservoirs = num_reservoirs
        self.num_plants = num_plants
        self.max_episode_steps = max_episode_steps
        self.continuous_management = continuous_management
        self.progressive_training_manager = progressive_training_manager
        self.use_fixed_connections = use_fixed_connections
        self.enable_optimizations = enable_optimizations

        # ==================== 数据加载 ====================
        self._load_environment_data()

        # ==================== 连接矩阵 ====================
        self.connections = self._generate_connections()
        # 计算距离矩阵用于成本优化
        self._calculate_distance_matrix()

        # ==================== 管理器初始化 ====================
        # 根据优化模式选择观测管理器
        if self.enable_optimizations:
            from onpolicy.envs.water_env.simplified_observation import SimplifiedObservationManager
            self.obs_manager = SimplifiedObservationManager(num_reservoirs, num_plants, self.connections)
            print("使用简化观测")
        else:
            self.obs_manager = StructuredObservationManager(num_reservoirs, num_plants, self.connections)
            print("使用标准观测")

        self.action_processor = ActionProcessor(num_reservoirs, num_plants, self.connections)
        self.continuous_state_manager = ContinuousStateManager(num_reservoirs, num_plants)

        # ==================== 奖励系统 ====================
        if self.enable_optimizations:
            from onpolicy.envs.water_env.simple_reward_system import SimpleRewardSystem
            self.reward_system = SimpleRewardSystem(
                n_reservoirs=num_reservoirs,
                n_plants=num_plants,
                max_episode_steps=max_episode_steps
            )
            print("使用简化奖励系统")
        else:
            self.reward_system = StabilizedMultiAgentRewardSystem(
                n_reservoirs=num_reservoirs,
                n_plants=num_plants,
                max_episode_steps=max_episode_steps,
                distance_matrix=self.distance_matrix, # 传递距离矩阵
                cost_factor=0.01  # 成本系数 p (可配置)
            )
            print("使用标准奖励系统")

        # 探索机制（优化模式下启用）
        if self.enable_optimizations:
            from onpolicy.envs.water_env.enhanced_exploration import EnhancedExplorationManager
            self.exploration_manager = EnhancedExplorationManager(
                n_reservoirs=num_reservoirs,
                n_plants=num_plants,
                max_episode_steps=max_episode_steps
            )
            print("增强探索机制已启用")
        else:
            self.exploration_manager = None

        # ==================== 状态初始化 ====================
        self.current_episode = 0
        self.current_global_episode = 0
        self.total_steps = 0
        self.current_hour = 0
        self.current_day = 0
        self.cumulative_days = 0
        self.season_progress = 0.0
        self._done = False

        # ==================== 历史记录 ====================
        self.reward_history = deque(maxlen=max_episode_steps)
        self.satisfaction_history = deque(maxlen=max_episode_steps)
        self.inflow_history = deque(maxlen=24)
        self.ecological_release_history = deque(maxlen=24)
        self.forced_spill_history = deque(maxlen=24)

        # ==================== 智能体定义 ====================
        self.agents = [f"reservoir_{i}" for i in range(num_reservoirs)] + \
                     [f"plant_{i}" for i in range(num_plants)]

        # ==================== 观测和动作空间 ====================
        self._setup_spaces()

        # ==================== 随机数生成器 ====================
        self.np_random = np.random.RandomState()

        # ==================== 渐进式训练支持 ====================
        self.use_simplified_actions = False
        self.simplified_action_processor = None

        # ==================== 其他状态 ====================
        self._prev_reward_state = None
        self._last_actual_supply = np.zeros(num_plants)
        self._last_target_ecological_flows = np.zeros(num_reservoirs)
        self._last_ecological_releases = np.zeros(num_reservoirs)
        self._last_apfd_deviation = 0.0

        print(f"WaterManagementEnv初始化完成")
        print(f"   水库数量: {num_reservoirs}, 水厂数量: {num_plants}")
        print(f"   使用固定连接: {use_fixed_connections}")
        print(f"   优化模式: {'启用' if enable_optimizations else '禁用'}")

    def _load_environment_data(self):
        """加载环境数据"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

        # 加载水库容量
        max_reservoir_file = os.path.join(data_dir, 'max_reservoir_capacity.csv')
        self.max_reservoir = self._load_data(max_reservoir_file, 'max_capacity')[:self.num_reservoirs]

        # 加载水厂需求
        plant_demand_file = os.path.join(data_dir, 'plant_demand.csv')
        self.max_plant = self._load_data(plant_demand_file, 'max_demand')[:self.num_plants]

        # 加载集雨面积
        self._load_basin_areas()

        # 计算衍生参数
        self.dead_capacity = self.max_reservoir * 0.15  # 死水位15%
        self.normal_capacity = self.max_reservoir * 0.85  # 正常水位85%
        self.normal_level = 0.85
        self.dead_level = 0.15

        # 水厂参数
        self.plant_storage_capacity = self.max_plant * 2.0  # 存储容量为日需求的2倍
        self.pipe_capacity = self.max_plant * 0.5  # 管道容量为日需求的50%

        # 初始化状态变量
        self.reservoirs = np.zeros(self.num_reservoirs)
        self.plants_demand = np.zeros(self.num_plants)
        self.plant_inventory = np.zeros(self.num_plants)

        print(f"环境数据加载完成")
        print(f"   水库容量: {self.max_reservoir.astype(int)}")
        print(f"   水厂需求: {self.max_plant.astype(int)}")
        print(f"   集雨面积: {self.reservoir_areas}")

    def _setup_spaces(self):
        """设置观测和动作空间"""
        # 观测空间
        max_obs_dim = self.obs_manager.max_obs_dim
        self.observation_space = {}
        for agent in self.agents:
            self.observation_space[agent] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(max_obs_dim,), dtype=np.float32
            )

        # 动作空间
        self.action_space = {}
        for i in range(self.num_reservoirs):
            agent_id = f"reservoir_{i}"
            self.action_space[agent_id] = spaces.Box(
                low=0.0, high=1.0,
                shape=(2 + self.num_plants,),
                dtype=np.float32
            )

        for i in range(self.num_plants):
            agent_id = f"plant_{i}"
            self.action_space[agent_id] = spaces.Box(
                low=0.0, high=2.0,
                shape=(3,),
                dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        """reset方法实现"""

        # ==================== 1. 基础设置 ====================
        if seed is not None:
            self.seed(seed)

        # Episode计数和状态重置
        self.current_episode += 1
        self.total_steps = 0
        self.current_hour = 0
        self.current_day = 0
        self._done = False

        # ==================== 2. 状态初始化 ====================
        # 使用确定性的初始状态分布而非完全随机
        self._initialize_deterministic_state()

        # ==================== 3. 历史清理 ====================
        self._reset_episode_data()

        # ==================== 4. 组件重置 ====================
        self._reset_subsystems()

        # ==================== 5. 观测生成 ====================
        observations = self._get_synchronized_observations()

        # ==================== 6. 验证和日志 ====================
        self._validate_reset_state()
        self._log_episode_start()

        return observations

    def _initialize_deterministic_state(self):
        """确定性的状态初始化"""
        # 水库：基于合理的运行水位
        base_levels = np.array([0.6, 0.55, 0.5, 0.5, 0.6, 0.55, 0.5, 0.6, 0.55, 0.5, 0.6, 0.55, 0.5, 0.6, 0.55])  # 不同水库的标准水位
        level_variance = 0.1  # 限制随机性
        noise = self.np_random.uniform(-level_variance, level_variance, self.num_reservoirs)
        safe_levels = np.clip(base_levels + noise, 0.3, 0.8)
        self.reservoirs = safe_levels * self.max_reservoir

        # 需求：基于季节性的合理需求
        base_demand_ratio = 0.4 + 0.2 * math.sin(2 * math.pi * self.season_progress)
        demand_noise = self.np_random.normal(0, 0.05, self.num_plants)
        demand_ratios = np.clip(base_demand_ratio + demand_noise, 0.3, 0.8)
        self.plants_demand = demand_ratios * self.max_plant

        # 季节：合理的季节起点
        if not hasattr(self, 'season_progress'):
            self.season_progress = self.np_random.uniform(0.0, 1.0)

        # 库存：标准初始库存
        self.plant_inventory = 0.4 * self.plant_storage_capacity

    def _reset_episode_data(self):
        """重置Episode级别的数据"""
        # Episode内历史
        self.reward_history = deque(maxlen=self.max_episode_steps)
        self.satisfaction_history = deque(maxlen=self.max_episode_steps)
        self.inflow_history = deque(maxlen=24)

        # Episode统计
        self.episode_rewards = []
        self.episode_success_count = 0

        # 清理观测历史
        if hasattr(self, 'obs_manager'):
            self.obs_manager.reset_history()

    def _reset_subsystems(self):
        """重置各个子系统"""
        # 奖励系统重置
        if hasattr(self.reward_system, 'reset_episode'):
            self.reward_system.reset_episode()

        # 动作处理器重置
        if hasattr(self.action_processor, 'reset'):
            self.action_processor.reset()

        # 其他组件重置
        self._prev_reward_state = None

    def _get_synchronized_observations(self):
        """获取同步的观测"""
        # 创建一致的状态快照
        env_state = {
            'reservoirs': self.reservoirs.copy(),
            'max_reservoir': self.max_reservoir.copy(),
            'plants_demand': self.plants_demand.copy(),
            'plant_inventory': self.plant_inventory.copy(),
            'current_hour': self.current_hour,
            'season_progress': self.season_progress,
            'total_steps': self.total_steps,
            'dead_capacity': getattr(self, 'dead_capacity', np.zeros(self.num_reservoirs)),
            'hourly_demand': self.plants_demand / 24.0,
            'actual_supply': np.zeros(self.num_plants),
            'max_plant': getattr(self, 'max_plant', np.ones(self.num_plants) * 10000),
            'plant_storage_capacity': getattr(self, 'plant_storage_capacity', np.ones(self.num_plants) * 5000),
            'target_ecological_flows': np.zeros(self.num_reservoirs),
            'ecological_releases': np.zeros(self.num_reservoirs)
        }

        # 基于观测管理器类型生成观测
        observations = {}
        for agent in self.agents:
            if hasattr(self.obs_manager, 'get_simplified_observation'):
                observations[agent] = self.obs_manager.get_simplified_observation(agent, env_state)
            else:
                observations[agent] = self.obs_manager.get_structured_observation(agent, env_state)

        return observations

    def _validate_reset_state(self):
        """验证重置后的状态合理性"""
        assert np.all(self.reservoirs >= 0), "水库水量不能为负"
        assert np.all(self.reservoirs <= self.max_reservoir), "水库水量不能超过容量"
        assert np.all(self.plants_demand >= 0), "需求不能为负"
        assert 0 <= self.season_progress <= 1, "季节进度必须在[0,1]范围内"

    def _log_episode_start(self):
        """记录Episode开始信息"""
        avg_level = np.mean(self.reservoirs / self.max_reservoir)
        total_demand = np.sum(self.plants_demand)

        if self.current_episode % 100 == 0:  # 每100个Episode记录一次
            print(f"Episode {self.current_episode} 开始")
            print(f"   平均水位: {avg_level:.2%}")
            print(f"   总需求: {total_demand:.0f} m³/天")
            print(f"   季节: {self.season_progress:.2f}")

    def step(self, actions):
        """步进方法"""

        if self.progressive_training_manager:
            self.progressive_training_manager.step()

        if hasattr(self, 'use_simplified_actions') and self.use_simplified_actions:
            if hasattr(self, 'simplified_action_processor'):
                try:
                    # 检查动作格式并转换
                    if not self._is_original_action_format(actions):
                        actions = self.simplified_action_processor.convert_simplified_to_original_actions(actions)
                except Exception as e:
                    print(f"简化动作转换失败: {e}")
                    actions = self._get_default_actions()

        self._advance_time()

        # 0. 动作预处理和验证
        try:
            hydrated_actions = self.action_processor.rehydrate_actions_fixed(actions)
            self._validate_actions(hydrated_actions)
        except Exception as e:
            print(f"动作格式错误: {e}, 将使用默认动作。")
            hydrated_actions = self._get_default_actions()

        state_snapshot = self._create_state_snapshot()

        # 1. 自然过程
        inflows = self._generate_improved_rainfall()
        self.reservoirs += inflows
        self.inflow_history.append(inflows)

        # 2. 人为调度决策
        self._update_plant_demands(hydrated_actions)
        water_released, actual_supply = self._enhanced_water_allocation(hydrated_actions)

        # 3. 生态流量处理和安全措施
        target_ecological_flows, ecological_releases = self._handle_ecological_flow()
        forced_spills = self._handle_flood_control()

        # 4. 物理约束
        self.reservoirs = np.clip(self.reservoirs, 0, self.max_reservoir)

        # 构建奖励状态
        reward_state = self._build_reward_state(state_snapshot, actual_supply, ecological_releases,
                                                target_ecological_flows, forced_spills)

        # 5.简化奖励计算
        is_terminal = self._check_done()
        prev_reward_state = getattr(self, '_prev_reward_state', None)

        #直接调用选定的奖励系统
        rewards, infos_rewards = self.reward_system.calculate_rewards(
            reward_state,
            hydrated_actions,
            self.total_steps,
            episode_progress=self.total_steps / self.max_episode_steps,
            is_terminal=is_terminal,
            prev_state=prev_reward_state
        )

        # 集成探索奖励（优化模式下）
        if self.enable_optimizations and self.exploration_manager is not None:
            try:
                exploration_rewards = self.exploration_manager.calculate_exploration_rewards(
                    hydrated_actions, reward_state
                )

                # 将探索奖励添加到主要奖励中
                for agent_id in rewards:
                    if agent_id in exploration_rewards:
                        rewards[agent_id] += exploration_rewards[agent_id]

                # 更新探索状态
                episode_rewards = {agent_id: rewards[agent_id] for agent_id in rewards}
                episode_performance = {'avg_reward': np.mean(list(rewards.values()))}
                exploration_state = self.exploration_manager.update_exploration_state(
                    self.current_episode, episode_rewards, episode_performance
                )

                # 将探索信息添加到info中
                if isinstance(infos_rewards, dict):
                    exploration_summary = self.exploration_manager.get_exploration_summary()
                    infos_rewards.update({
                        'exploration_state': exploration_state,
                        'exploration_summary': exploration_summary,
                        'exploration_rewards': exploration_rewards
                    })

            except Exception as e:
                print(f"探索机制计算失败: {e}")

        # 保存当前状态作为下一步的前一状态
        self._prev_reward_state = reward_state.copy()

        # 6. 更新历史记录
        self._update_history_and_stats(rewards, actual_supply, ecological_releases, target_ecological_flows)

        # 7. 检查终止条件
        self._done = is_terminal
        terminations = {agent: self._done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # 8. 生成观测
        observations = self._get_all_observations()

        # 9. 改进的信息字典构建
        infos = self._build_enhanced_info_dict(infos_rewards, inflows, water_released, actual_supply,
                                               ecological_releases, target_ecological_flows, forced_spills)

        # 每24步（一天）打印一次总结信息
        if self.current_hour == 23:  # 在每天的最后一小时打印
            avg_reward = np.mean(list(rewards.values())) if rewards else 0.0
            avg_reservoir_level = np.mean(self.reservoirs / self.max_reservoir)

            # 供水满足率
            hourly_demand = self.plants_demand / 24.0
            total_supply = np.sum(actual_supply)
            total_demand = np.sum(hourly_demand)
            supply_satisfaction_rate = min(total_supply / (total_demand + 1e-8), 1.0) if total_demand > 0 else 1.0

            avg_apfd_deviation = getattr(self, '_last_apfd_deviation', 0.0)

            print(f"Day {self.cumulative_days} | "
                  f"Reward: {avg_reward:.3f} | "
                  f"Reservoir: {avg_reservoir_level:.2%} | "
                  f"Supply: {supply_satisfaction_rate:.2%} | "
                  f"APFD Dev: {avg_apfd_deviation:.3f}")

            try:
                import wandb
                if wandb.run is not None:
                    daily_data = {
                        "daily/day": int(self.cumulative_days),
                        "daily/avg_reward": float(avg_reward),
                        "daily/supply_satisfaction": float(supply_satisfaction_rate),
                        "daily/reservoir_level": float(avg_reservoir_level),
                        "daily/apfd_deviation": float(avg_apfd_deviation),
                        "daily/exploration_reward": float(
                            infos_rewards.get('reward_components', {}).get('exploration', 0.0)),
                        "global_step": int(self.total_steps)
                    }
                    wandb.log(daily_data)
                    # print(f"每日wandb记录: Day {self.cumulative_days}")
            except Exception as e:
                print(f" 每日wandb记录失败: {e}")

            # 如果有探索信息，也打印出来
            if 'exploration_active' in infos_rewards and infos_rewards['exploration_active']:
                exploration_reward = infos_rewards.get('reward_components', {}).get('exploration', 0.0)
                print(f"Exploration Active | Exploration Reward: {exploration_reward:.3f}")

        return observations, rewards, terminations, truncations, infos

    def _advance_time(self):
        """时间推进逻辑"""
        self.current_hour = (self.current_hour + 1) % 24
        self.total_steps += 1

        # 新的一天
        if self.current_hour == 0:
            self.current_day += 1
            self.cumulative_days += 1

            if self.continuous_management:
                self.season_progress = (self.cumulative_days / 30.0) % 1.0
            else:
                self.season_progress = (self.current_day / 30.0) % 1.0

    def _create_state_snapshot(self):
        """创建状态快照用于奖励计算"""
        return {
            'reservoirs': self.reservoirs.copy(),
            'max_reservoir': self.max_reservoir.copy(),
            'dead_capacity': self.dead_capacity.copy(),
            'plants_demand': self.plants_demand.copy(),
            'max_plant': self.max_plant.copy(),
            'plant_inventory': self.plant_inventory.copy(),
            'plant_storage_capacity': self.plant_storage_capacity.copy(),
            'current_hour': self.current_hour,
            'season_progress': self.season_progress
        }

    def _build_reward_state(self, state_snapshot, actual_supply, ecological_releases, target_ecological_flows, forced_spills):
        """构建奖励计算所需的状态字典"""
        hourly_demand = state_snapshot['plants_demand'] / 24.0

        return {
            'actual_supply': actual_supply,
            'hourly_demand': hourly_demand,
            'reservoirs': state_snapshot['reservoirs'],
            'max_reservoir': state_snapshot['max_reservoir'],
            'dead_capacity': state_snapshot['dead_capacity'],
            'plant_inventory': state_snapshot['plant_inventory'],
            'plant_storage_capacity': state_snapshot['plant_storage_capacity'],
            'ecological_releases': ecological_releases,
            'target_ecological_flows': target_ecological_flows,
            'current_hour': state_snapshot['current_hour'],
            'season_progress': state_snapshot['season_progress'],
            'forced_spills': forced_spills
        }

    def _get_all_observations(self):
        """获取所有智能体的观测"""
        # 构建环境状态用于观测计算
        env_state = {
            'reservoirs': self.reservoirs,
            'max_reservoir': self.max_reservoir,
            'dead_capacity': self.dead_capacity,
            'hourly_demand': self.plants_demand / 24.0,
            'actual_supply': getattr(self, '_last_actual_supply', np.zeros(self.num_plants)),
            'max_plant': self.max_plant,
            'plant_inventory': self.plant_inventory,
            'plant_storage_capacity': self.plant_storage_capacity,
            'current_hour': self.current_hour,
            'season_progress': self.season_progress,
            'target_ecological_flows': getattr(self, '_last_target_ecological_flows', np.zeros(self.num_reservoirs)),
            'ecological_releases': getattr(self, '_last_ecological_releases', np.zeros(self.num_reservoirs))
        }

        # 更新历史缓存
        self.obs_manager.update_history(env_state)

        # 生成观测
        observations = {}
        for agent in self.agents:
            if hasattr(self.obs_manager, 'get_simplified_observation'):
                # 使用简化观测管理器
                observations[agent] = self.obs_manager.get_simplified_observation(agent, env_state)
            else:
                # 使用标准观测管理器
                observations[agent] = self.obs_manager.get_structured_observation(agent, env_state)

        return observations

    def _validate_actions(self, actions):
        """验证动作格式的正确性"""
        for agent_id, action in actions.items():
            if not isinstance(action, dict):
                raise ValueError(f"动作必须是字典格式: {agent_id}")

            if agent_id.startswith("reservoir"):
                required_keys = ['total_release_ratio', 'allocation_weights', 'emergency_release']
                if not all(key in action for key in required_keys):
                    raise ValueError(f"水库动作 {agent_id} 缺少必要键值")
            elif agent_id.startswith("plant"):
                required_keys = ['demand_adjustment', 'priority_level', 'storage_strategy']
                if not all(key in action for key in required_keys):
                    raise ValueError(f"水厂动作 {agent_id} 缺少必要键值")

    def _get_default_actions(self):
        """获取默认动作"""
        default_actions = {}
        for i in range(self.num_reservoirs):
            agent_id = f"reservoir_{i}"
            default_actions[agent_id] = {
                'total_release_ratio': np.array([0.1]),
                'allocation_weights': np.ones(self.num_plants) / self.num_plants,
                'emergency_release': 0
            }
        for i in range(self.num_plants):
            agent_id = f"plant_{i}"
            default_actions[agent_id] = {
                'demand_adjustment': np.array([1.0]),
                'priority_level': 1,
                'storage_strategy': np.array([0.5])
            }
        return default_actions

    def _update_plant_demands(self, hydrated_actions):
        """根据智能体动作更新水厂需求"""
        base_daily_demands = self.generate_plant_demands()
        adjusted_demands = base_daily_demands.copy()

        for i in range(self.num_plants):
            agent_id = f"plant_{i}"
            if agent_id in hydrated_actions:
                action = hydrated_actions[agent_id]
                demand_factor = action.get('demand_adjustment', [1.0])[0]
                adjusted_demands[i] = base_daily_demands[i] * demand_factor

        self.plants_demand = np.clip(adjusted_demands, 0, self.max_plant * 2.0)

    def _enhanced_water_allocation(self, actions):
        """水量分配"""


        hourly_demand = self.plants_demand / 24.0
        remaining_demand = hourly_demand.copy()
        actual_supply = np.zeros(self.num_plants)
        water_released = np.zeros(self.num_reservoirs)

        # 收集水厂优先级信息
        plant_priorities = np.ones(self.num_plants)
        for i in range(self.num_plants):
            agent_id = f"plant_{i}"
            if agent_id in actions:
                action = actions[agent_id]
                if isinstance(action, dict) and 'priority_level' in action:
                    plant_priorities[i] = 0.5 + 0.5 * action['priority_level']

        # 按随机顺序处理水库
        for i in self.np_random.permutation(self.num_reservoirs):
            agent_id = f"reservoir_{i}"
            if agent_id not in actions:
                continue

            action = actions[agent_id]
            if not isinstance(action, dict):
                continue

            # 死水位检查
            current_level_ratio = self.reservoirs[i] / self.max_reservoir[i]

            # 分级供水策略
            if current_level_ratio <= 0.2:  # ≤20%
                # 死水位：停止所有非紧急供水
                continue
            elif current_level_ratio <= 0.3:  # 20%-30%
                # 紧急水位：严格限制供水
                max_supply_ratio = 0.2  # 最多释放20%可用水量
            elif current_level_ratio <= 0.4:  # 30%-40%
                # 警戒水位：限制供水
                max_supply_ratio = 0.5  # 最多释放50%可用水量
            else:
                # 正常水位：无限制
                max_supply_ratio = 1.0

            # 紧急释放处理
            emergency_release_action = action.get('emergency_release', 0)
            if emergency_release_action == 1 and self.reservoirs[i] > self.normal_capacity[i]:
                spill_amount = (self.reservoirs[i] - self.normal_capacity[i]) * 0.5
                spill_amount = min(spill_amount, self.reservoirs[i] - self.dead_capacity[i])
                spill_amount = max(0, spill_amount)
                if spill_amount > 0:
                    self.reservoirs[i] -= spill_amount

            # 正常供水分配
            connected_plants = np.where(self.connections[i, :])[0]
            if len(connected_plants) == 0:
                continue

            available_water = max(0, self.reservoirs[i] - self.dead_capacity[i])
            if available_water <= 0:
                continue

            # 计算释放量 - 应用供水限制
            total_release_ratio = action.get('total_release_ratio', [0.0])[0]
            constrained_ratio = total_release_ratio * max_supply_ratio
            total_release_intent = available_water * constrained_ratio

            if total_release_intent <= 0:
                continue

            # 使用全维度分配权重，但只考虑连接的水厂
            allocation_weights = action.get('allocation_weights', np.ones(self.num_plants))

            # 只对连接的水厂进行分配
            connected_weights = allocation_weights[connected_plants]
            if np.sum(connected_weights) > 0:
                connected_weights = connected_weights / np.sum(connected_weights)
            else:
                connected_weights = np.ones(len(connected_plants)) / len(connected_plants)

            # 结合优先级
            plant_priorities_subset = plant_priorities[connected_plants]
            combined_weights = connected_weights * plant_priorities_subset
            if np.sum(combined_weights) > 0:
                combined_weights = combined_weights / np.sum(combined_weights)
            else:
                combined_weights = np.ones(len(connected_plants)) / len(connected_plants)

            # 计算实际分配
            desired_supply = total_release_intent * combined_weights
            demand_limited_supply = np.minimum(desired_supply, remaining_demand[connected_plants])
            final_supply_to_plants = np.minimum(demand_limited_supply, self.pipe_capacity)

            # 执行分配
            actual_total_supply_release = np.sum(final_supply_to_plants)
            if actual_total_supply_release > 0:
                self.reservoirs[i] -= actual_total_supply_release
                water_released[i] += actual_total_supply_release

                for k, plant_idx in enumerate(connected_plants):
                    supply_amount = final_supply_to_plants[k]
                    if supply_amount > 0:
                        actual_supply[plant_idx] += supply_amount
                        remaining_demand[plant_idx] = max(0, remaining_demand[plant_idx] - supply_amount)

        # 保存供水信息用于观测
        self._last_actual_supply = actual_supply.copy()

        return water_released, actual_supply

    def _handle_ecological_flow(self):
        """生态流量处理 - 添加偏差计算"""
        target_ecological_flows = self._calculate_current_target_flow()
        available_for_eco = np.maximum(0, self.reservoirs - self.dead_capacity)

        # 紧急状态检查
        reservoir_levels = self.reservoirs / self.max_reservoir
        system_emergency = np.any(reservoir_levels < 0.25)
        system_crisis = np.any(reservoir_levels < 0.20)

        # 减少噪音
        if system_crisis:
            # 严重危机：暂停生态流量
            target_ecological_flows = target_ecological_flows * 0.1
            # 只在每天输出一次，而不是每小时
            if self.current_hour == 0:
                crisis_reservoirs = np.where(reservoir_levels < 0.20)[0]
                print(f" 系统危机：水库{crisis_reservoirs}水位<20%，大幅减少生态流量")
        elif system_emergency:
            # 紧急状态：减少50%生态流量
            target_ecological_flows = target_ecological_flows * 0.5
            # 只在每天输出一次
            if self.current_hour == 0:
                emergency_reservoirs = np.where(reservoir_levels < 0.25)[0]
                print(f" 系统紧急：水库{emergency_reservoirs}水位<25%，减少生态流量")

        # 实际释放考虑能力限制
        max_release_capacity = self.max_reservoir * 0.01  # 最大释放能力

        desired_releases = np.minimum(target_ecological_flows, available_for_eco)
        actual_releases = np.minimum(desired_releases, max_release_capacity)

        self.reservoirs -= actual_releases

        # 正确计算APFD偏差
        ecological_deviations = []
        for i in range(self.num_reservoirs):
            if target_ecological_flows[i] > 0:
                deviation = abs(actual_releases[i] - target_ecological_flows[i]) / target_ecological_flows[i]
                ecological_deviations.append(deviation)

        avg_apfd_deviation = np.mean(ecological_deviations) if ecological_deviations else 0.0

        # 保存信息用于观测和日志
        self._last_target_ecological_flows = target_ecological_flows.copy()
        self._last_ecological_releases = actual_releases.copy()
        self._last_apfd_deviation = avg_apfd_deviation

        return target_ecological_flows, actual_releases

    def _handle_flood_control(self):
        """处理防洪泄洪"""
        forced_spills = np.zeros(self.num_reservoirs)
        if self.current_hour % 6 == 0:
            for i in range(self.num_reservoirs):
                reservoir_ratio = self.reservoirs[i] / self.max_reservoir[i]
                if reservoir_ratio > self.normal_level:
                    release_factor = (reservoir_ratio - self.normal_level) * 0.5
                    release_amount = self.reservoirs[i] * release_factor
                    release_amount = min(release_amount, self.reservoirs[i] - self.dead_capacity[i])
                    release_amount = max(0, release_amount)

                    self.reservoirs[i] -= release_amount
                    forced_spills[i] += release_amount

        self.forced_spill_history.append(forced_spills)
        return forced_spills

    def _update_history_and_stats(self, rewards, actual_supply, ecological_releases, target_ecological_flows):
        """更新历史记录和统计数据"""
        # 奖励历史
        avg_reward = np.mean(list(rewards.values())) if rewards else 0.0
        self.reward_history.append(avg_reward)

        # 供水满足率历史
        hourly_demand = self.plants_demand / 24.0
        total_supply = np.sum(actual_supply)
        total_demand = np.sum(hourly_demand)
        supply_satisfaction_rate = min(total_supply / (total_demand + 1e-8), 1.0) if total_demand > 0 else 1.0
        self.satisfaction_history.append(supply_satisfaction_rate)

        # 生态流量历史
        self.ecological_release_history.append(ecological_releases)

    def _check_done(self):
        """检查环境是否结束"""
        if self.total_steps >= self.max_episode_steps:
            return True

        # 检查提前终止条件
        if np.all(self.reservoirs < 0):
            print(" 提前终止: 所有水库水量均低于0")
            return True

        if len(self.satisfaction_history) >= 12:
            recent_satisfaction = np.array(list(self.satisfaction_history)[-12:])
            if np.mean(recent_satisfaction) < 0.05:
                print("提前终止: 系统供水满足率持续过低")
                return True

        return False

    def _build_enhanced_info_dict(self, infos_rewards, inflows, water_released, actual_supply, ecological_releases,
                                  target_ecological_flows, forced_spills):
        """构建增强的信息字典 - 完整整合奖励系统信息"""

        # 计算实时指标
        reservoir_levels = self.reservoirs / self.max_reservoir
        safe_reservoirs = np.sum((reservoir_levels >= self.dead_level) & (reservoir_levels <= self.normal_level))
        calculated_reservoir_safety = safe_reservoirs / self.num_reservoirs
        supply_satisfaction_rate = self.satisfaction_history[-1] if self.satisfaction_history else 0.0
        avg_reward = self.reward_history[-1] if self.reward_history else 0.0

        # 基础信息
        base_info = {
            "state": self.state,
            "supply_satisfaction_rate": supply_satisfaction_rate,
            "avg_reservoir_level": calculated_reservoir_safety,
            "system_efficiency": avg_reward,
            "current_hour": self.current_hour,
            "current_day": self.current_day,
            "total_steps": self.total_steps,
            "cumulative_days": self.cumulative_days,
            "inflows": inflows.tolist(),
            "water_released": water_released.tolist(),
            "actual_supply": actual_supply.tolist(),
            "ecological_releases": ecological_releases.tolist(),
            "target_ecological_flows": target_ecological_flows.tolist(),
            "forced_spills": forced_spills.tolist(),
        }

        infos = {}

        # 为每个智能体构建信息
        for agent in self.agents:
            info = base_info.copy()

            #添加来自奖励系统的详细信息
            if agent in infos_rewards:
                agent_reward_info = infos_rewards[agent]
                info.update(agent_reward_info)

            infos[agent] = info

        # 添加来自奖励系统的全局信息
        if infos_rewards and isinstance(infos_rewards, dict):
            global_info = {}

            # 提取全局信息键
            global_keys = [
                'phase', 'reward_components', 'convergence_metrics',
                'normalizer_stats', 'exploration_active', 'diversity_score'
            ]

            for key in global_keys:
                if key in infos_rewards:
                    global_info[key] = infos_rewards[key]

            # 将全局信息添加到每个智能体
            for agent in self.agents:
                if agent in infos:
                    infos[agent].update(global_info)

        return infos

    def enable_training_optimizations(self):
        """启用训练优化模式"""
        print("正在启用训练优化模式...")

        # 1. 切换到简化奖励系统
        from onpolicy.envs.water_env.simple_reward_system import SimpleRewardSystem
        old_reward_system = self.reward_system
        self.reward_system = SimpleRewardSystem(
            n_reservoirs=self.num_reservoirs,
            n_plants=self.num_plants,
            max_episode_steps=self.max_episode_steps
        )

        # 2. 切换到简化观测管理器
        from onpolicy.envs.water_env.simplified_observation import SimplifiedObservationManager
        old_obs_manager = self.obs_manager
        self.obs_manager = SimplifiedObservationManager(
            self.num_reservoirs, self.num_plants, self.connections
        )

        # 3. 启用探索机制
        if not hasattr(self, 'exploration_manager') or self.exploration_manager is None:
            from onpolicy.envs.water_env.enhanced_exploration import EnhancedExplorationManager
            self.exploration_manager = EnhancedExplorationManager(
                n_reservoirs=self.num_reservoirs,
                n_plants=self.num_plants,
                max_episode_steps=self.max_episode_steps
            )

        # 4. 更新观测空间
        self._setup_optimized_spaces()

        # 5. 设置优化标志
        self.enable_optimizations = True

        print("训练优化模式已启用")
        print(f"   - 奖励系统: {type(self.reward_system).__name__}")
        print(f"   - 观测管理器: {type(self.obs_manager).__name__}")
        print(f"   - 探索机制: {'YES' if self.exploration_manager else 'NO'}")
        print(f"   - 观测维度: {self.obs_manager.max_obs_dim}")

        return True

    def _setup_optimized_spaces(self):
        """设置优化的观测空间"""
        # 更新观测空间以匹配简化的观测管理器
        max_obs_dim = self.obs_manager.max_obs_dim

        self.observation_space = {}
        for agent in self.agents:
            self.observation_space[agent] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(max_obs_dim,), dtype=np.float32
            )

        print(f"观测空间已更新: {max_obs_dim}维")

    def get_optimization_status(self):
        """获取优化状态信息"""
        return {
            'optimizations_enabled': self.enable_optimizations,
            'reward_system_type': type(self.reward_system).__name__,
            'obs_manager_type': type(self.obs_manager).__name__,
            'exploration_enabled': self.exploration_manager is not None,
            'obs_dimension': getattr(self.obs_manager, 'max_obs_dim', 30),
            'components': {
                'simple_rewards': isinstance(self.reward_system, type(None)) or 'Simple' in type(self.reward_system).__name__,
                'simple_observations': hasattr(self.obs_manager, 'get_simplified_observation'),
                'enhanced_exploration': self.exploration_manager is not None
            }
        }

    # 辅助方法
    def generate_plant_demands(self):
        """水厂需求生成 - 基于可持续供水能力"""
        base_demands = self.max_plant.copy()
        season_factor = 0.5 + 0.5 * math.sin(2 * math.pi * self.season_progress)

        # 季节性需求变化
        if season_factor < 0.3:  # 冬季
            seasonal_adjustment = 0.8  # 减少20%
        elif season_factor > 0.7:  # 夏季
            seasonal_adjustment = 1.2  # 增加20%
        else:  # 春秋
            seasonal_adjustment = 1.0

        random_adjustments = 1.0 + self.np_random.normal(0, 0.05, self.num_plants)
        new_demands = base_demands * seasonal_adjustment * random_adjustments


        # 计算系统的可持续供水能力
        total_reservoir_capacity = np.sum(self.max_reservoir)
        annual_inflow_capacity = total_reservoir_capacity * 1.5  # 年入流能力
        daily_inflow_capacity = annual_inflow_capacity / 365

        # 生态流量预留（按30%计算）
        sustainable_supply_capacity = daily_inflow_capacity * 0.7  # 70%可用于供水

        # 确保需求不超过可持续供水能力的90%（留10%安全余量）
        total_demand = np.sum(new_demands)
        max_sustainable_demand = sustainable_supply_capacity * 0.9

        if total_demand > max_sustainable_demand:
            scale_factor = max_sustainable_demand / total_demand
            new_demands = new_demands * scale_factor

            # 调试信息
            if self.current_hour == 0 and self.current_day % 30 == 0:  # 每30天打印一次
                print(f"需求控制: 总需求{total_demand:.0f} > 可持续上限{max_sustainable_demand:.0f}")
                print(f"   缩放系数: {scale_factor:.3f}")

        return new_demands


    def _generate_improved_rainfall(self):
        """MBLRP模型：调整系数以达到合理入流量"""
        inflow = np.zeros(self.num_reservoirs)

        # 保持MBLRP的物理季节性模式
        season_factor = 0.3 + 0.7 * math.sin(2 * math.pi * self.season_progress)

        # 物理真实的降雨模式
        if season_factor < 0.4:  # 枯水期
            mean_seasonal_rain = 0.8 + 0.6 * season_factor  # 0.8-1.04
        elif season_factor > 0.8:  # 丰水期
            mean_seasonal_rain = 1.2 + 0.8 * season_factor  # 1.84-1.96
        else:  # 平水期
            mean_seasonal_rain = 1.0 + 0.5 * season_factor  # 1.2-1.4

        # 保持适度随机性（气象变化）
        noise = self.np_random.normal(0, 0.2, self.num_reservoirs)
        rainfall = np.maximum(0.2, mean_seasonal_rain + noise)

        # MBLRP模型计算 - 调整系数达到合理入流量
        for i in range(self.num_reservoirs):
            # 径流系数基于土壤类型和季节
            runoff_coefficient = 0.15 + 0.25 * season_factor  # 0.15-0.4

            # 基础径流（地表径流）
            surface_runoff = rainfall[i] * self.reservoir_areas[i] * runoff_coefficient * 130

            # 地下水补给（慢速释放）
            groundwater_inflow = self.reservoir_areas[i] * 0.5 * (1 + 0.3 * season_factor)

            # 总入流 = 地表径流 + 地下水补给
            total_inflow = surface_runoff + groundwater_inflow
            inflow[i] = total_inflow

        return inflow

    def _calculate_current_target_flow(self):
        """基于实际需求的生态流量"""

        # 使用实际需求而不是最大需求
        current_daily_demand = np.sum(self.plants_demand)
        hourly_demand = current_daily_demand / 24.0

        # 生态流量 = 实际需求的25-35%
        base_eco_ratio = 0.30  # 基础30%

        # 季节性调整：丰水期增加，枯水期减少
        season_factor = 0.3 + 0.7 * math.sin(2 * math.pi * self.season_progress)
        seasonal_eco_ratio = base_eco_ratio * (0.8 + 0.4 * season_factor)

        # 水位影响：低水位时减少生态流量
        reservoir_levels = self.reservoirs / self.max_reservoir
        avg_level = np.mean(reservoir_levels)

        if avg_level < 0.25:
            level_factor = 0.6  # 危机：减少40%
        elif avg_level < 0.4:
            level_factor = 0.8  # 警戒：减少20%
        else:
            level_factor = 1.0  # 正常：不减少

        # 计算总生态流量需求
        total_eco_flow = hourly_demand * seasonal_eco_ratio * level_factor

        # 按库容比例分配到各水库
        reservoir_shares = self.max_reservoir / np.sum(self.max_reservoir)
        target_flows = total_eco_flow * reservoir_shares

        # 设定合理范围
        min_flow_per_reservoir = hourly_demand * 0.05  # 最少5%
        max_flow_per_reservoir = hourly_demand * 0.15  # 最多15%
        target_flows = np.clip(target_flows, min_flow_per_reservoir, max_flow_per_reservoir)

        return target_flows

    def _load_basin_areas(self):
        """按照数据加载集雨面积"""
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            basin_areas_file = os.path.join(data_dir, 'basin_areas.csv')

            if os.path.exists(basin_areas_file):
                data = pd.read_csv(basin_areas_file)

                if 'area' in data.columns:
                    areas = data['area'].values

                    if len(areas) >= self.num_reservoirs:
                        self.reservoir_areas = areas[:self.num_reservoirs]
                    else:
                        self.reservoir_areas = np.append(areas, [areas[-1]] * (self.num_reservoirs - len(areas)))

                    print(f"使用csv文件数据 - 集雨面积: {self.reservoir_areas} km²")

                    # 使用MBLRP模型预估平均入流量
                    avg_rainfall = 1.3
                    avg_runoff_coeff = 0.275
                    avg_season_factor = 0.65

                    estimated_inflows = []
                    for area in self.reservoir_areas:
                        surface_runoff = avg_rainfall * area * avg_runoff_coeff * 130
                        groundwater_inflow = area * 0.5 * (1 + 0.3 * avg_season_factor)
                        total_inflow = surface_runoff + groundwater_inflow
                        estimated_inflows.append(total_inflow)

                    total_estimated = sum(estimated_inflows)
                    print(f" 基于调整系数后的预估入流量:")
                    print(f"   各水库: {[f'{x:.1f}' for x in estimated_inflows]} m³/小时")
                    print(f"   总计: {total_estimated:.1f} m³/小时 ({total_estimated*24:.0f} m³/天)")

                else:
                    raise ValueError("CSV文件缺少'area'列")

            else:
                raise FileNotFoundError("basin_areas.csv文件不存在")

        except Exception as e:
            print(f"加载集水区面积失败: {e}")
            # 如果文件读取失败，使用文件中显示的默认值
            self.reservoir_areas = np.array([22.06, 4, 5, 12])[:self.num_reservoirs]
            print(f" 使用备用数据 - 集雨面积: {self.reservoir_areas} km²")

    def _load_data(self, file_path, column_name):
        """从CSV文件加载数据"""
        try:
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                if column_name in data.columns:
                    return data[column_name].values
                else:
                    return self._get_default_data(column_name)
            else:
                return self._get_default_data(column_name)
        except Exception as e:
            print(f"加载数据时出错：{e}")
            return self._get_default_data(column_name)

    def _get_default_data(self, column_name):
        """获取默认数据"""
        if column_name == 'max_capacity':
            return np.array([59190000, 20000000, 20000000, 5700000, 5000000,
                             1000000, 14700000, 10000000, 10000000,
                             10000000, 10000000, 1000000])
        elif column_name == 'max_demand':
            return np.array([8000, 7500, 8200])
        else:
            return np.array([1000] * 10)

    def _generate_connections(self):
        """连接矩阵"""
        # 使用固定连接或随机生成
        if self.use_fixed_connections and self.num_reservoirs == 15 and self.num_plants == 6:
            # 使用预定义的固定连接（15个水库连接到6个水厂）
            fixed_connections = np.array([
                [True, False, False, False, False, False],  # reservoir_0 -> plant_0
                [False, False, True, False, False, False],  # reservoir_1 -> plant_2
                [False, False, True, False, False, False],  # reservoir_2 -> plant_2
                [False, True, False, False, False, False],  # reservoir_3 -> plant_1
                [False, True, False, False, False, False],  # reservoir_4 -> plant_1
                [False, True, False, False, False, False],  # reservoir_5 -> plant_1
                [False, False, False, True, False, False],
                [False, False, False, False, True, False],
                [True, False, False, False, False, False],  # reservoir_8 -> plant_0
                [False, False, False, False, False, True],
                [True, False, False, False, False, False],  # reservoir_10 -> plant_0
                [True, False, False, False, False, False],  # reservoir_11 -> plant_0
                [True, False, False, False, False, False],  # reservoir_12 -> plant_0
                [True, False, False, False, False, False],  # reservoir_13 -> plant_0
                [False, True, False, False, False, False],  # reservoir_14 -> plant_1
            ])
            print(" 使用固定连接矩阵 (15水库->6水厂)")
        else:
            # 生成随机连接矩阵
            connections = np.random.choice([True, False],
                                         size=(self.num_reservoirs, self.num_plants),
                                         p=[0.3, 0.7])

            # 确保每个水厂至少有一个连接
            for j in range(self.num_plants):
                if not connections[:, j].any():
                    reservoir_idx = np.random.randint(0, self.num_reservoirs)
                    connections[reservoir_idx, j] = True

            print(f" 生成随机连接矩阵 ({self.num_reservoirs}水库->{self.num_plants}水厂)")

        return connections

    def seed(self, seed=None):
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def render(self, mode="human"):
        """渲染环境"""
        if mode == "human":
            print("=" * 60)
            print(f"Step: {self.total_steps}, Hour: {self.current_hour}, Day: {self.current_day}")
            print(f"Season: {self.season_progress:.2%}")

            reservoir_levels = self.reservoirs / self.max_reservoir
            print(f" L: {np.mean(reservoir_levels):.2%}")

            if self.satisfaction_history:
                print(f"Supply Satisfaction: {self.satisfaction_history[-1]:.2%}")
            print("=" * 60)

    def close(self):
        """关闭环境"""
        pass

    @property
    def observation_spaces(self):
        """向后兼容性属性"""
        return self.observation_space

    @property
    def action_spaces(self):
        """向后兼容性属性"""
        return self.action_space

    @property
    def state(self):
        """返回环境的全局状态"""
        state = []

        # 水库状态
        reservoir_ratios = self.reservoirs / self.max_reservoir
        state.extend(reservoir_ratios)

        # 水厂状态
        plant_demand_ratios = self.plants_demand / self.max_plant
        state.extend(plant_demand_ratios)

        # 全局统计信息
        total_reservoir = np.sum(self.reservoirs) / np.sum(self.max_reservoir)
        total_demand = np.sum(self.plants_demand) / np.sum(self.max_plant)
        state.extend([total_reservoir, total_demand])

        # 时间信息
        state.append(self.current_hour / 24.0)
        state.append(self.current_day / 365.0)
        state.append(self.season_progress)

        # 入流信息
        if self.inflow_history:
            current_inflows = self.inflow_history[-1]
            max_inflows = self.max_reservoir * 0.1
            normalized_inflows = current_inflows / (max_inflows + 1e-8)
            state.extend(normalized_inflows)

            # 入流趋势
            if len(self.inflow_history) >= 6:
                past_inflows = np.mean([inflow for inflow in list(self.inflow_history)[-6:]], axis=0)
                inflow_trends = (current_inflows - past_inflows) / (max_inflows + 1e-8)
                state.extend(inflow_trends)
            else:
                state.extend(np.zeros(self.num_reservoirs))
        else:
            state.extend(np.zeros(self.num_reservoirs * 2))

        return np.array(state, dtype=np.float32)

    def get_reward_system_status(self):
        """获取奖励系统状态"""
        if hasattr(self.reward_system, 'convergence_monitor'):
            convergence_metrics = self.reward_system.convergence_monitor.get_metrics()
            curriculum_suggestion = self.reward_system.get_curriculum_suggestion()

            return {
                'convergence_metrics': convergence_metrics,
                'curriculum_suggestion': curriculum_suggestion,
                'normalizer_stats': self.reward_system.reward_normalizer.get_stats(),
                'current_episode': getattr(self, 'current_global_episode', 0)
            }
        else:
            return {'status': 'reward_system_unavailable'}

    def print_reward_analysis(self, detailed=False):
        """打印奖励分析"""
        status = self.get_reward_system_status()

        print("=" * 60)
        print("Reward System Analysis")
        print("=" * 60)

        if 'convergence_metrics' in status:
            metrics = status['convergence_metrics']
            print(f"Convergence Metrics:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"  {key}: trend={value.get('trend', 0):.3f}, stability={value.get('stability', 0):.3f}")

        if 'curriculum_suggestion' in status:
            suggestion = status['curriculum_suggestion']
            print(f"\n Curriculum Suggestion: {suggestion.get('suggestion', 'continue')}")
            print(f"   Reason: {suggestion.get('reason', 'unknown')}")

        print("=" * 60)

    def get_enhanced_reward_analysis(self):
        """增强的奖励分析"""
        if not hasattr(self.reward_system, 'typed_action_manager'):
            return {'status': 'enhanced_system_not_available'}

        return {
            'exploration_summary': self.reward_system.exploration_recorder.get_exploration_summary(),
            'agent_profiles': self.reward_system.exploration_recorder.get_agent_exploration_profiles(),
            'action_diversity': self.reward_system.typed_action_manager.get_action_diversity_metrics(),
            'action_stats': self.reward_system.typed_action_manager.get_action_stats(),
            'convergence_metrics': self.reward_system.convergence_monitor.get_metrics()
        }

    def print_enhanced_reward_analysis(self, detailed=False):
        """打印增强的奖励分析 - 在 WaterManagementEnv 类中添加"""
        analysis = self.get_enhanced_reward_analysis()

        if 'status' in analysis:
            print(f" {analysis['status']}")
            return

        print("=" * 80)
        print("Enhanced Reward System Analysis")
        print("=" * 80)

        # 探索状态总结
        exploration = analysis['exploration_summary']
        print(f"Exploration State:")
        print(f"   Phase: {exploration['current_state']['phase']}")
        print(f"   Active: {exploration['current_state']['exploration_active']}")
        print(f"   Episode: {exploration['current_state']['episode_count']}")
        print(f"   Effectiveness: {exploration['current_state']['exploration_effectiveness']:.3f}")

        # 探索指标
        metrics = exploration['metrics']
        print(f"\n Exploration Metrics:")
        print(f"   Discovery Rate: {metrics['discovery_rate']:.3f}")
        print(f"   Convergence Speed: {metrics['convergence_speed']:.3f}")
        print(f"   Stability Progress: {metrics['stability_progression']:.3f}")
        print(f"   Reward Improvement: {metrics['reward_improvement']:.3f}")

        # 动作统计
        action_stats = analysis['action_stats']
        print(f"\n Action Statistics:")
        print(f"   Reservoir Actions: {action_stats['reservoir']['total_actions']}")
        print(f"   Emergency Releases: {action_stats['reservoir']['emergency_releases']}")
        print(f"   Plant Actions: {action_stats['plant']['total_actions']}")
        print(f"   Demand Adjustments: {action_stats['plant']['demand_adjustments']}")

        if detailed:
            # 智能体档案
            profiles = analysis['agent_profiles']
            print(f"\n Agent Profiles (Top 5):")
            for i, (agent_id, profile) in enumerate(list(profiles.items())[:5]):
                print(f"   {agent_id}: {profile['dominant_strategy']}, "
                      f"Risk/Adapt: {profile['risk_level']:.2f}, "
                      f"Performance: {profile['recent_performance']:.3f}")

        print("=" * 80)

    def _is_original_action_format(self, actions):
        """检查动作是否为原始格式"""
        if not isinstance(actions, dict):
            return False

        for agent_id, action in actions.items():
            if not isinstance(action, dict):
                return False

            if agent_id.startswith('reservoir_'):
                required_keys = {'total_release_ratio', 'allocation_weights', 'emergency_release'}
                if not all(key in action for key in required_keys):
                    return False
            elif agent_id.startswith('plant_'):
                required_keys = {'demand_adjustment', 'priority_level', 'storage_strategy'}
                if not all(key in action for key in required_keys):
                    return False
        
        return True

    def _calculate_distance_matrix(self):
        """计算水库与水厂之间的距离矩阵（用于成本优化）"""
        self.distance_matrix = np.zeros((self.num_reservoirs, self.num_plants))
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

        try:
            reservoir_coords_file = os.path.join(data_dir, 'reservoir_coordinates.csv')
            plant_coords_file = os.path.join(data_dir, 'plant_coordinates.csv')

            # 读取水库坐标
            res_coords = pd.read_csv(reservoir_coords_file)

            # 读取水厂坐标
            plant_coords = pd.read_csv(plant_coords_file)

            print(f" 坐标加载: {len(res_coords)}个水库, {len(plant_coords)}个水厂")

            # 计算距离矩阵
            for i in range(self.num_reservoirs):
                for j in range(self.num_plants):
                    if i < len(res_coords) and j < len(plant_coords):
                        res_lon, res_lat = res_coords.iloc[i]['longitude'], res_coords.iloc[i]['latitude']
                        plant_lon, plant_lat = plant_coords.iloc[j]['longitude'], plant_coords.iloc[j]['latitude']

                        # 使用哈弗赛因公式计算地理距离
                        distance = self._haversine_distance(res_lon, res_lat, plant_lon, plant_lat)
                        self.distance_matrix[i, j] = distance
                    else:
                        # 当智能体数量多于坐标文件中的数量时的备用方案
                        self.distance_matrix[i, j] = abs(i - j) * 10.0 + 5.0

            print(f" 距离矩阵计算完成 (单位: km)")
            print(f"   形状: {self.distance_matrix.shape}")
            print(f"   距离范围: {self.distance_matrix.min():.2f} - {self.distance_matrix.max():.2f} km")

        except Exception as e:
            print(f" 计算距离矩阵失败: {e}")
            print("️ 使用基于索引的默认距离")
            # 创建备用距离矩阵
            for i in range(self.num_reservoirs):
                for j in range(self.num_plants):
                    self.distance_matrix[i, j] = abs(i - j) * 10.0 + 5.0

    def _haversine_distance(self, lon1, lat1, lon2, lat2):
        """使用哈弗赛因公式计算两点之间的地理距离（单位：公里）"""
        # 地球半径，单位公里
        R = 6371.0

        # 将度转换为弧度
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        # 计算差值
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # 哈弗赛因公式
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        distance = R * c

        return distance