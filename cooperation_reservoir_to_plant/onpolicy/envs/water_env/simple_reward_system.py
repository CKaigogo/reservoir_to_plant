"""
简化奖励系统
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple, List


class SimpleRewardSystem:
    """简化的奖励"""
    
    def __init__(self, n_reservoirs=12, n_plants=3, max_episode_steps=96):
        self.n_reservoirs = n_reservoirs
        self.n_plants = n_plants
        self.max_episode_steps = max_episode_steps
        
        # 简化的奖励权重
        self.reward_weights = {
            'supply_satisfaction': 1.0,  # 降低从2.0
            'reservoir_safety': 1.0,     # 降低从5.0  
            'ecological_balance': 0.5,   # 降低从1.0
            'stability_bonus': 0.2       # 降低从2.0
        }

        self.reward_smoother = {agent_id: deque(maxlen=5) for agent_id in self._get_all_agent_ids()}
        self.performance_history = deque(maxlen=20)
        
        # 奖励范围控制
        self.min_reward = -2.0
        self.max_reward = 5.0
        
        print("简化奖励系统已初始化")
        # print(f"   - 奖励范围: [{self.min_reward}, {self.max_reward}]")
        # print(f"   - 权重简化: {self.reward_weights}")

    def calculate_rewards(self, state: Dict, actions: Dict, current_step: int,
                         episode_progress: float = None, is_terminal: bool = False,
                         prev_state: Dict = None) -> Tuple[Dict, Dict]:
        """简化的奖励计算 - 专注于稳定性"""
        
        try:
            # 1. 计算核心奖励组件
            supply_rewards = self._calculate_simple_supply_rewards(state)
            safety_rewards = self._calculate_simple_safety_rewards(state)
            ecological_rewards = self._calculate_simple_ecological_rewards(state)
            stability_rewards = self._calculate_simple_stability_rewards()
            
            # 2. 合并奖励（线性组合）
            raw_rewards = {}
            for agent_id in self._get_all_agent_ids():
                raw_reward = (
                    supply_rewards.get(agent_id, 0.0) * self.reward_weights['supply_satisfaction'] +
                    safety_rewards.get(agent_id, 0.0) * self.reward_weights['reservoir_safety'] +
                    ecological_rewards.get(agent_id, 0.0) * self.reward_weights['ecological_balance'] +
                    stability_rewards.get(agent_id, 0.0) * self.reward_weights['stability_bonus']
                )
                raw_rewards[agent_id] = raw_reward
            
            # 3.应用奖励平滑和范围限制
            final_rewards = self._apply_smoothing_and_clipping(raw_rewards)
            
            # 4. 更新历史记录
            self._update_performance_history(final_rewards, state)
            
            # 5. 构建简化的信息字典
            info_dict = self._build_simple_info_dict(
                supply_rewards, safety_rewards, ecological_rewards, 
                stability_rewards, state, final_rewards
            )
            
            return final_rewards, info_dict
            
        except Exception as e:
            print(f" 简化奖励计算错误: {e}")
            # 返回安全的默认奖励
            return self._get_safe_default_rewards()

    def _calculate_simple_supply_rewards(self, state: Dict) -> Dict:
        """简化的供水奖励 - 线性计算"""
        rewards = {}
        
        total_supply = np.sum(state.get('actual_supply', [0]))
        total_demand = np.sum(state.get('hourly_demand', [1]))
        
        # 简单的满足率计算
        satisfaction_rate = min(total_supply / (total_demand + 1e-8), 1.0)
        
        # 线性奖励，避免非线性函数
        base_supply_reward = satisfaction_rate * 2.0 - 1.0  # 范围[-1, 1]
        
        # 分配给所有智能体
        for agent_id in self._get_all_agent_ids():
            rewards[agent_id] = base_supply_reward
            
        return rewards

    def _calculate_simple_safety_rewards(self, state: Dict) -> Dict:
        """简化的安全奖励 - 线性分段函数"""
        rewards = {}
        
        reservoir_levels = state.get('reservoirs', np.zeros(self.n_reservoirs))
        max_reservoirs = state.get('max_reservoir', np.ones(self.n_reservoirs))
        
        for i in range(self.n_reservoirs):
            agent_id = f"reservoir_{i}"
            level_ratio = reservoir_levels[i] / (max_reservoirs[i] + 1e-8)
            
            # 分段线性函数
            if level_ratio < 0.2:      # 危险水位
                safety_reward = -1.0
            elif level_ratio < 0.3:    # 警戒水位  
                safety_reward = -0.5
            elif level_ratio < 0.8:    # 正常水位
                safety_reward = 0.5
            else:                      # 高水位
                safety_reward = 0.0
                
            rewards[agent_id] = safety_reward
            
        # 水厂获得平均安全奖励
        avg_safety = np.mean(list(rewards.values())) if rewards else 0.0
        for i in range(self.n_plants):
            agent_id = f"plant_{i}"
            rewards[agent_id] = avg_safety
            
        return rewards

    def _calculate_simple_ecological_rewards(self, state: Dict) -> Dict:
        """简化的生态奖励"""
        rewards = {}
        
        ecological_releases = state.get('ecological_releases', np.zeros(self.n_reservoirs))
        target_flows = state.get('target_ecological_flows', np.zeros(self.n_reservoirs))
        
        # 简单的达标奖励
        total_compliance = 0.0
        for i in range(self.n_reservoirs):
            if target_flows[i] > 0:
                compliance = min(ecological_releases[i] / (target_flows[i] + 1e-8), 1.0)
                total_compliance += compliance
        
        avg_compliance = total_compliance / max(self.n_reservoirs, 1)
        base_eco_reward = (avg_compliance - 0.5) * 0.5  # 范围[-0.25, 0.25]
        
        # 分配给所有智能体
        for agent_id in self._get_all_agent_ids():
            rewards[agent_id] = base_eco_reward
            
        return rewards

    def _calculate_simple_stability_rewards(self) -> Dict:
        """简化的稳定性奖励"""
        rewards = {}
        
        # 🎯 基于历史性能的稳定性奖励
        if len(self.performance_history) >= 5:
            recent_performance = list(self.performance_history)[-5:]
            stability = 1.0 / (1.0 + np.std(recent_performance))
            stability_reward = (stability - 0.5) * 0.2  # 小幅奖励
        else:
            stability_reward = 0.0
            
        # 分配给所有智能体
        for agent_id in self._get_all_agent_ids():
            rewards[agent_id] = stability_reward
            
        return rewards

    def _apply_smoothing_and_clipping(self, raw_rewards: Dict) -> Dict:
        """应用平滑和范围限制"""
        final_rewards = {}
        
        for agent_id, raw_reward in raw_rewards.items():
            # 1. 添加到平滑器
            self.reward_smoother[agent_id].append(raw_reward)
            
            # 2. 计算平滑奖励（移动平均）
            if len(self.reward_smoother[agent_id]) >= 3:
                smoothed_reward = np.mean(list(self.reward_smoother[agent_id]))
            else:
                smoothed_reward = raw_reward
            
            # 3.严格的范围限制
            clipped_reward = np.clip(smoothed_reward, self.min_reward, self.max_reward)
            
            final_rewards[agent_id] = clipped_reward
            
        return final_rewards

    def _update_performance_history(self, rewards: Dict, state: Dict):
        """更新性能历史"""
        avg_reward = np.mean(list(rewards.values())) if rewards else 0.0
        self.performance_history.append(avg_reward)

    def _get_all_agent_ids(self) -> List[str]:
        """获取所有智能体ID"""
        agent_ids = []
        for i in range(self.n_reservoirs):
            agent_ids.append(f"reservoir_{i}")
        for i in range(self.n_plants):
            agent_ids.append(f"plant_{i}")
        return agent_ids

    def _build_simple_info_dict(self, supply_rewards, safety_rewards, ecological_rewards, 
                               stability_rewards, state, final_rewards):
        """构建简化的信息字典"""
        
        # 计算全局指标
        total_supply = np.sum(state.get('actual_supply', [0]))
        total_demand = np.sum(state.get('hourly_demand', [1]))
        supply_satisfaction = min(total_supply / (total_demand + 1e-8), 1.0)
        
        reservoir_levels = state.get('reservoirs', np.zeros(self.n_reservoirs))
        max_reservoirs = state.get('max_reservoir', np.ones(self.n_reservoirs))
        avg_reservoir_level = np.mean(reservoir_levels / (max_reservoirs + 1e-8))
        
        # 简化的信息字典
        base_info = {
            'supply_satisfaction_rate': supply_satisfaction,
            'avg_reservoir_level': avg_reservoir_level,
            'reward_components': {
                'supply': np.mean(list(supply_rewards.values())) if supply_rewards else 0.0,
                'safety': np.mean(list(safety_rewards.values())) if safety_rewards else 0.0,
                'ecological': np.mean(list(ecological_rewards.values())) if ecological_rewards else 0.0,
                'stability': np.mean(list(stability_rewards.values())) if stability_rewards else 0.0
            },
            'reward_range': f"[{self.min_reward}, {self.max_reward}]",
            'system_type': 'simplified_reward_system'
        }
        
        # 为每个智能体复制基础信息
        info_dict = {}
        for agent_id in self._get_all_agent_ids():
            agent_info = base_info.copy()
            agent_info['total_reward'] = final_rewards.get(agent_id, 0.0)
            info_dict[agent_id] = agent_info
            
        return info_dict

    def _get_safe_default_rewards(self):
        """获取安全的默认奖励"""
        default_rewards = {agent_id: 0.0 for agent_id in self._get_all_agent_ids()}
        default_info = {agent_id: {'total_reward': 0.0, 'system_type': 'simplified_reward_system'} 
                       for agent_id in self._get_all_agent_ids()}
        return default_rewards, default_info 