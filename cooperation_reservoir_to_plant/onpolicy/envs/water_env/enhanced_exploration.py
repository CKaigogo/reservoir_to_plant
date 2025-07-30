"""
探索机制
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple


class EnhancedExplorationManager:
    """探索管理器 - 探索策略"""
    
    def __init__(self, n_reservoirs=4, n_plants=1, max_episode_steps=168):
        self.n_reservoirs = n_reservoirs
        self.n_plants = n_plants
        self.max_episode_steps = max_episode_steps
        
        # 🎯 探索参数
        self.exploration_config = {
            'initial_exploration_rate': 0.3,
            'min_exploration_rate': 0.05,
            'decay_episodes': 500,
            'diversity_threshold': 0.1,
            'reward_threshold': 0.5
        }
        

        self.exploration_state = {
            'current_episode': 0,
            'exploration_active': True,
            'exploration_rate': self.exploration_config['initial_exploration_rate'],
            'phase': 'exploration'  # exploration, transition, exploitation
        }
        
        #动作历史记录
        self.action_history = {agent_id: deque(maxlen=100) for agent_id in self._get_all_agent_ids()}
        self.performance_history = deque(maxlen=50)
        self.diversity_history = deque(maxlen=30)
        
        # 探索奖励配置
        self.exploration_rewards = {
            'diversity_bonus': 0.1,      # 多样性奖励
            'novelty_bonus': 0.05,       # 新颖性奖励  
            'progress_bonus': 0.02       # 进步奖励
        }
        
        print(" 探索机制已初始化")
        # print(f"   - 初始探索率: {self.exploration_config['initial_exploration_rate']}")
        # print(f"   - 多样性阈值: {self.exploration_config['diversity_threshold']}")

    def update_exploration_state(self, episode: int, episode_rewards: Dict, episode_performance: Dict):
        """更新探索状态"""
        self.exploration_state['current_episode'] = episode
        
        # 更新历史记录
        avg_reward = np.mean(list(episode_rewards.values())) if episode_rewards else 0.0
        self.performance_history.append(avg_reward)

        current_diversity = self._calculate_current_diversity()
        self.diversity_history.append(current_diversity)
        
        # 更新探索率（指数衰减）
        decay_factor = max(0, (self.exploration_config['decay_episodes'] - episode) / self.exploration_config['decay_episodes'])
        self.exploration_state['exploration_rate'] = (
            self.exploration_config['min_exploration_rate'] + 
            (self.exploration_config['initial_exploration_rate'] - self.exploration_config['min_exploration_rate']) * decay_factor
        )

        self._update_exploration_phase()
        
        return self.exploration_state

    def calculate_exploration_rewards(self, actions: Dict, state: Dict) -> Dict:
        """计算探索奖励"""
        exploration_rewards = {}
        
        for agent_id in self._get_all_agent_ids():
            total_exploration_reward = 0.0
            
            # 1. 多样性奖励
            diversity_reward = self._calculate_diversity_reward(agent_id, actions.get(agent_id, {}))
            total_exploration_reward += diversity_reward * self.exploration_rewards['diversity_bonus']
            
            # 2. 新颖性奖励
            novelty_reward = self._calculate_novelty_reward(agent_id, actions.get(agent_id, {}))
            total_exploration_reward += novelty_reward * self.exploration_rewards['novelty_bonus']
            
            # 3. 进步奖励
            progress_reward = self._calculate_progress_reward()
            total_exploration_reward += progress_reward * self.exploration_rewards['progress_bonus']
            
            # 4. 应用探索权重
            final_exploration_reward = total_exploration_reward * self.exploration_state['exploration_rate']
            
            exploration_rewards[agent_id] = final_exploration_reward
            
        # 记录动作历史
        self._record_actions(actions)
        
        return exploration_rewards

    def _calculate_diversity_reward(self, agent_id: str, action: Dict) -> float:
        """计算多样性奖励"""
        if not action or len(self.action_history[agent_id]) < 5:
            return 0.0
        
        # 提取动作特征
        current_features = self._extract_action_features(agent_id, action)
        
        # 与历史动作比较
        recent_actions = list(self.action_history[agent_id])[-10:]
        if not recent_actions:
            return 1.0  # 首次动作给予满分
        
        # 计算与最近动作的差异度
        diversity_scores = []
        for past_features in recent_actions:
            if past_features is not None:
                # 计算欧几里得距离
                diff = np.linalg.norm(current_features - past_features)
                diversity_scores.append(diff)
        
        if diversity_scores:
            avg_diversity = np.mean(diversity_scores)
            # 归一化到[0, 1]范围
            normalized_diversity = min(avg_diversity / 2.0, 1.0)
            return normalized_diversity
        
        return 0.0

    def _calculate_novelty_reward(self, agent_id: str, action: Dict) -> float:
        """计算新颖性奖励"""
        if not action:
            return 0.0
        
        current_features = self._extract_action_features(agent_id, action)
        
        # 检查是否为全新的动作组合
        all_history = list(self.action_history[agent_id])
        if not all_history:
            return 1.0  # 首次动作
        
        # 找到最相似的历史动作
        min_distance = float('inf')
        for past_features in all_history:
            if past_features is not None:
                distance = np.linalg.norm(current_features - past_features)
                min_distance = min(min_distance, distance)
        
        # 如果距离足够大，认为是新颖动作
        novelty_threshold = 0.3
        if min_distance > novelty_threshold:
            return min(min_distance / 1.0, 1.0)  # 归一化
        
        return 0.0

    def _calculate_progress_reward(self) -> float:
        """计算进步奖励"""
        if len(self.performance_history) < 10:
            return 0.0
        
        # 比较最近和早期的性能
        recent_performance = np.mean(list(self.performance_history)[-5:])
        early_performance = np.mean(list(self.performance_history)[:5])
        
        improvement = recent_performance - early_performance
        # 归一化改进程度
        progress_reward = np.tanh(improvement)
        
        return max(progress_reward, 0.0)

    def _extract_action_features(self, agent_id: str, action: Dict) -> np.ndarray:
        """提取动作特征向量"""
        if agent_id.startswith('reservoir_'):
            # 水库动作特征
            release_ratio = action.get('total_release_ratio', [0.1])[0]
            allocation_weights = action.get('allocation_weights', [1.0])
            emergency_release = action.get('emergency_release', 0)
            
            # 构造特征向量
            features = [release_ratio, emergency_release]
            if isinstance(allocation_weights, (list, np.ndarray)):
                features.extend(allocation_weights[:3])
            else:
                features.append(allocation_weights)
            
            return np.array(features[:5])
        
        else:  # plant
            # 水厂动作特征
            demand_adjustment = action.get('demand_adjustment', [1.0])[0]
            priority_level = action.get('priority_level', 1)
            storage_strategy = action.get('storage_strategy', [0.5])[0]
            
            return np.array([demand_adjustment, priority_level, storage_strategy, 0.0, 0.0])  # 填充到5维

    def _calculate_current_diversity(self) -> float:
        """计算当前系统的整体多样性"""
        all_diversities = []
        
        for agent_id in self._get_all_agent_ids():
            agent_history = list(self.action_history[agent_id])
            if len(agent_history) >= 5:
                recent_features = agent_history[-5:]
                # 计算内部多样性
                diversity_scores = []
                for i in range(len(recent_features)):
                    for j in range(i+1, len(recent_features)):
                        if recent_features[i] is not None and recent_features[j] is not None:
                            diff = np.linalg.norm(recent_features[i] - recent_features[j])
                            diversity_scores.append(diff)
                
                if diversity_scores:
                    agent_diversity = np.mean(diversity_scores)
                    all_diversities.append(agent_diversity)
        
        return np.mean(all_diversities) if all_diversities else 0.0

    def _update_exploration_phase(self):
        """更新探索阶段"""
        episode = self.exploration_state['current_episode']
        exploration_rate = self.exploration_state['exploration_rate']
        
        # 根据episode数量和性能确定阶段
        if episode < 200:
            self.exploration_state['phase'] = 'exploration'
            self.exploration_state['exploration_active'] = True
        elif episode < 800:
            self.exploration_state['phase'] = 'transition'
            self.exploration_state['exploration_active'] = True
        else:
            # 检查是否需要继续探索
            if len(self.performance_history) >= 20:
                recent_performance = list(self.performance_history)[-10:]
                performance_std = np.std(recent_performance)
                
                if performance_std < 0.1:  # 性能稳定，减少探索
                    self.exploration_state['phase'] = 'exploitation'
                    self.exploration_state['exploration_active'] = False
                else:
                    self.exploration_state['phase'] = 'transition'
                    self.exploration_state['exploration_active'] = True
            else:
                self.exploration_state['phase'] = 'transition'
                self.exploration_state['exploration_active'] = True

    def _record_actions(self, actions: Dict):
        """记录动作历史"""
        for agent_id, action in actions.items():
            if agent_id in self.action_history:
                features = self._extract_action_features(agent_id, action)
                self.action_history[agent_id].append(features)

    def _get_all_agent_ids(self) -> List[str]:
        """获取所有智能体ID"""
        agent_ids = []
        for i in range(self.n_reservoirs):
            agent_ids.append(f"reservoir_{i}")
        for i in range(self.n_plants):
            agent_ids.append(f"plant_{i}")
        return agent_ids

    def get_exploration_summary(self) -> Dict:
        """获取探索状态摘要"""
        return {
            'current_state': self.exploration_state.copy(),
            'metrics': {
                'current_diversity': self._calculate_current_diversity(),
                'avg_performance': np.mean(list(self.performance_history)) if self.performance_history else 0.0,
                'exploration_effectiveness': self._calculate_exploration_effectiveness()
            }
        }

    def _calculate_exploration_effectiveness(self) -> float:
        """计算探索有效性"""
        if len(self.diversity_history) < 5 or len(self.performance_history) < 5:
            return 0.5  # 默认值
        
        # 多样性趋势
        recent_diversity = np.mean(list(self.diversity_history)[-5:])
        early_diversity = np.mean(list(self.diversity_history)[:5])
        diversity_trend = recent_diversity - early_diversity
        
        # 性能趋势
        recent_performance = np.mean(list(self.performance_history)[-10:])
        early_performance = np.mean(list(self.performance_history)[:10])
        performance_trend = recent_performance - early_performance
        
        # 综合评估：多样性与性能提升
        effectiveness = 0.6 * performance_trend + 0.4 * diversity_trend
        return np.clip(effectiveness, 0.0, 1.0) 