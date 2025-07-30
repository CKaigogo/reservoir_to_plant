"""
奖励系统 - 应用势函数奖励塑形和稳定性改进
基于PBRS理论
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Tuple, List, Any, Optional
import math


class ExplorationRecorder:
    """探索记录器 - 追踪和奖励探索行为"""
    
    def __init__(self, n_reservoirs=12, n_plants=3, max_episode_steps=168):
        self.n_reservoirs = n_reservoirs
        self.n_plants = n_plants
        self.max_episode_steps = max_episode_steps
        
        # 探索状态
        self.exploration_active = True
        self.exploration_phase = 'initial'
        self.episode_count = 0
        self.exploration_effectiveness = 0.0
        
        # 动作多样性追踪
        self.action_history = defaultdict(lambda: deque(maxlen=50))
        self.action_diversity_scores = defaultdict(float)
        
        # 状态探索追踪
        self.state_visit_counts = defaultdict(int)
        self.recent_discoveries = deque(maxlen=20)
        
        # 探索奖励参数
        self.exploration_bonus_scale = 1.0
        self.diversity_threshold = 0.3
        self.discovery_bonus = 0.5
        
        # 探索指标
        self.exploration_metrics = {
            'discovery_rate': 0.0,
            'convergence_speed': 0.0,
            'stability_progression': 0.0,
            'reward_improvement': 0.0
        }
        
        print("探索记录器已初始化")
    
    def record_episode_start(self):
        """记录新Episode开始"""
        self.episode_count += 1
        
        # 更新探索阶段
        if self.episode_count < 100:
            self.exploration_phase = 'initial'
            self.exploration_active = True
        elif self.episode_count < 500:
            self.exploration_phase = 'expansion'
            self.exploration_active = True
        elif self.episode_count < 1000:
            self.exploration_phase = 'refinement'
            self.exploration_active = True
        else:
            self.exploration_phase = 'exploitation'
            self.exploration_active = False

        # 强制终止探索的条件
        if self.episode_count > 1500:  # 硬上限：1500个episode后强制停止探索
            self.exploration_active = False
            self.exploration_phase = 'forced_exploitation'
    
    def record_actions(self, actions: Dict, state: Dict) -> Dict:
        """记录动作并计算探索奖励"""
        exploration_rewards = {}
        
        if not self.exploration_active:
            # 探索阶段结束，返回0奖励
            return {agent_id: 0.0 for agent_id in self._get_all_agent_ids()}
        
        # 计算动作多样性
        for agent_id, action in actions.items():
            if isinstance(action, dict):
                # 将动作转换为特征向量
                action_features = self._extract_action_features(action, agent_id)
                
                # 记录动作历史
                self.action_history[agent_id].append(action_features)
                
                # 计算多样性分数
                diversity_score = self._calculate_action_diversity(agent_id)
                self.action_diversity_scores[agent_id] = diversity_score
                
                # 计算探索奖励
                exploration_reward = self._calculate_exploration_reward(agent_id, diversity_score, state)
                exploration_rewards[agent_id] = exploration_reward
        
        # 记录状态访问
        state_key = self._get_state_key(state)
        self.state_visit_counts[state_key] += 1
        
        # 发现新状态的奖励
        if self.state_visit_counts[state_key] == 1:
            self.recent_discoveries.append(state_key)
            # 给所有智能体发现奖励
            for agent_id in exploration_rewards:
                exploration_rewards[agent_id] += self.discovery_bonus
        
        return exploration_rewards
    
    def _extract_action_features(self, action: Dict, agent_id: str) -> np.ndarray:
        """提取动作特征向量"""
        if agent_id.startswith('reservoir'):
            # 水库动作特征
            release_ratio = action.get('total_release_ratio', [0.0])[0]
            weights = action.get('allocation_weights', [1.0])
            emergency = action.get('emergency_release', 0)
            
            features = [release_ratio, np.mean(weights), np.std(weights), float(emergency)]
        else:
            # 水厂动作特征
            demand_adj = action.get('demand_adjustment', [1.0])[0]
            priority = action.get('priority_level', 1)
            storage = action.get('storage_strategy', [0.5])[0]
            
            features = [demand_adj, float(priority), storage]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_action_diversity(self, agent_id: str) -> float:
        """计算动作多样性分数"""
        if len(self.action_history[agent_id]) < 2:
            return 0.0
        
        # 计算最近动作的标准差
        recent_actions = list(self.action_history[agent_id])[-10:]
        action_matrix = np.array(recent_actions)
        
        # 计算每个特征的标准差
        feature_stds = np.std(action_matrix, axis=0)
        
        # 平均标准差作为多样性分数
        diversity_score = np.mean(feature_stds)
        
        return diversity_score
    
    def _calculate_exploration_reward(self, agent_id: str, diversity_score: float, state: Dict) -> float:
        """计算探索奖励"""
        if not self.exploration_active:
            return 0.0
        
        # 基础多样性奖励
        diversity_bonus = 0.0
        if diversity_score > self.diversity_threshold:
            diversity_bonus = self.exploration_bonus_scale * (diversity_score - self.diversity_threshold)
        
        # 基于当前状态的探索奖励
        state_bonus = 0.0
        if agent_id.startswith('reservoir'):
            # 水库探索奖励：鼓励在不同水位下尝试不同策略
            reservoir_idx = int(agent_id.split('_')[1])
            if reservoir_idx < len(state['reservoirs']):
                level_ratio = state['reservoirs'][reservoir_idx] / state['max_reservoir'][reservoir_idx]
                # 在危险水位下探索更有价值
                if level_ratio < 0.4:
                    state_bonus = 0.2
                elif level_ratio > 0.8:
                    state_bonus = 0.1
        
        total_exploration_reward = diversity_bonus + state_bonus
        
        # 根据探索阶段调整奖励
        phase_multiplier = {
            'initial': 1.0,
            'expansion': 0.8,
            'refinement': 0.5,
            'exploitation': 0.1
        }
        
        total_exploration_reward *= phase_multiplier.get(self.exploration_phase, 0.5)
        
        return total_exploration_reward
    
    def _get_state_key(self, state: Dict) -> str:
        """获取状态的唯一键"""
        # 简化状态表示
        reservoir_levels = state['reservoirs'] / state['max_reservoir']
        supply_ratios = state['actual_supply'] / (state['hourly_demand'] + 1e-8)
        
        # 离散化状态
        discrete_levels = np.round(reservoir_levels, 1)
        discrete_supply = np.round(supply_ratios, 1)
        
        return f"r{discrete_levels.tolist()}_s{discrete_supply.tolist()}"
    
    def update_exploration_effectiveness(self, rewards: Dict, state: Dict):
        """更新探索效果"""
        # 简单的探索效果评估
        avg_reward = np.mean(list(rewards.values()))
        diversity_score = np.mean(list(self.action_diversity_scores.values()))
        
        # 结合奖励和多样性
        self.exploration_effectiveness = 0.7 * avg_reward + 0.3 * diversity_score
        
        # 更新探索指标
        self.exploration_metrics['discovery_rate'] = len(self.recent_discoveries) / 20.0
        self.exploration_metrics['convergence_speed'] = min(1.0, self.episode_count / 1000.0)
        self.exploration_metrics['stability_progression'] = self.exploration_effectiveness
        self.exploration_metrics['reward_improvement'] = max(0, avg_reward)
    
    def get_exploration_summary(self) -> Dict:
        """获取探索总结"""
        return {
            'current_state': {
                'phase': self.exploration_phase,
                'exploration_active': self.exploration_active,
                'episode_count': self.episode_count,
                'exploration_effectiveness': self.exploration_effectiveness
            },
            'metrics': self.exploration_metrics
        }
    
    def get_agent_exploration_profiles(self) -> Dict:
        """获取智能体探索档案"""
        profiles = {}
        
        for agent_id in self._get_all_agent_ids():
            diversity_score = self.action_diversity_scores.get(agent_id, 0.0)
            
            # 根据多样性分数分类策略
            if diversity_score > 0.5:
                strategy = 'explorer'
            elif diversity_score > 0.3:
                strategy = 'balanced'
            else:
                strategy = 'conservative'
            
            profiles[agent_id] = {
                'dominant_strategy': strategy,
                'risk_level': diversity_score,
                'recent_performance': self.exploration_effectiveness
            }
        
        return profiles
    
    def _get_all_agent_ids(self) -> List[str]:
        """获取所有智能体ID"""
        agent_ids = []
        for i in range(self.n_reservoirs):
            agent_ids.append(f"reservoir_{i}")
        for i in range(self.n_plants):
            agent_ids.append(f"plant_{i}")
        return agent_ids


class PotentialBasedRewardShaper:
    """势函数奖励塑形器 - 解决收敛问题的关键组件"""

    def __init__(self, n_reservoirs, n_plants, gamma=0.995):
        self.n_reservoirs = n_reservoirs
        self.n_plants = n_plants
        self.gamma = gamma

        # 势函数历史
        self.prev_potentials = {}

        # 多层次势函数设计
        self.potential_weights = {
            'supply_potential': 2.0,  # 供水满足势能
            'safety_potential': 6.0,  # 安全势能
            'ecological_potential': 2.0  # 生态势能
        }

    def calculate_potential_shaping(self, state: Dict, prev_state: Dict = None) -> Dict:
        """计算势函数奖励塑形"""
        current_potentials = self._calculate_potentials(state)

        if prev_state is None or not self.prev_potentials:
            # 初始状态，无塑形奖励
            shaping_rewards = {agent_id: 0.0 for agent_id in self._get_all_agent_ids()}
        else:
            # PBRS公式：F(s,a,s') = γ*Φ(s') - Φ(s)
            shaping_rewards = {}
            for agent_id in self._get_all_agent_ids():
                current_phi = current_potentials.get(agent_id, 0.0)
                prev_phi = self.prev_potentials.get(agent_id, 0.0)
                shaping_rewards[agent_id] = self.gamma * current_phi - prev_phi

        # 保存当前势能
        self.prev_potentials = current_potentials.copy()

        return shaping_rewards

    def _calculate_potentials(self, state: Dict) -> Dict:
        """计算多层次势函数"""
        potentials = {}

        # 1. 供水满足势能
        supply_potentials = self._calculate_supply_potential(state)

        # 2. 安全势能
        safety_potentials = self._calculate_safety_potential(state)

        # 3. 生态势能
        ecological_potentials = self._calculate_ecological_potential(state)

        # 合并势能
        for agent_id in self._get_all_agent_ids():
            total_potential = (
                    supply_potentials.get(agent_id, 0.0) * self.potential_weights['supply_potential'] +
                    safety_potentials.get(agent_id, 0.0) * self.potential_weights['safety_potential'] +
                    ecological_potentials.get(agent_id, 0.0) * self.potential_weights['ecological_potential']
            )
            potentials[agent_id] = total_potential

        return potentials

    def _calculate_supply_potential(self, state: Dict) -> Dict:
        """计算供水满足势函数 - 非线性势能"""
        potentials = {}

        # 全局供水满足率
        total_supply = np.sum(state['actual_supply'])
        total_demand = np.sum(state['hourly_demand'])
        global_satisfaction = min(total_supply / (total_demand + 1e-8), 1.0)

        # 使用sigmoid势函数-提供更平滑的梯度
        def sigmoid_potential(x, steepness=4.0, midpoint=0.8):
            """S型势函数，在目标附近提供强梯度"""
            return 2.0 / (1.0 + np.exp(-steepness * (x - midpoint))) - 1.0

        base_potential = sigmoid_potential(global_satisfaction)

        # 为不同类型智能体分配势能
        for i in range(self.n_reservoirs):
            agent_id = f"reservoir_{i}"
            # 水库基于支持能力的势能
            reservoir_level = state['reservoirs'][i] / state['max_reservoir'][i]
            level_potential = sigmoid_potential(reservoir_level, steepness=3.0, midpoint=0.6)
            potentials[agent_id] = base_potential * 0.7 + level_potential * 0.3

        for i in range(self.n_plants):
            agent_id = f"plant_{i}"
            # 水厂基于需求满足的势能
            individual_satisfaction = min(
                state['actual_supply'][i] / (state['hourly_demand'][i] + 1e-8), 1.0
            )
            individual_potential = sigmoid_potential(individual_satisfaction)
            potentials[agent_id] = base_potential * 0.5 + individual_potential * 0.5

        return potentials

    def _calculate_safety_potential(self, state: Dict) -> Dict:
        """计算安全势函数"""
        potentials = {}

        def safety_potential_function(level_ratio):
            """安全势函数：在0.25-0.75范围内为正，其他为负"""
            optimal_center = 0.5
            safe_width = 0.25

            if 0.25 <= level_ratio <= 0.75:  # 扩大安全范围
                distance_from_center = abs(level_ratio - optimal_center)
                return 3.0 - (distance_from_center / safe_width) ** 2
            else:
                # 危险区域的势能急剧下降
                if level_ratio < 0.25:
                    return -5.0 * (0.25 - level_ratio) / 0.3
                else:
                    return -5.0 * (level_ratio - 0.75) / 0.2

        # 计算每个水库的安全势能
        reservoir_safety_potentials = []
        for i in range(self.n_reservoirs):
            level_ratio = state['reservoirs'][i] / state['max_reservoir'][i]
            safety_pot = safety_potential_function(level_ratio)
            reservoir_safety_potentials.append(safety_pot)

        # 全局安全势能
        global_safety_potential = np.mean(reservoir_safety_potentials)

        # 分配给智能体
        for i in range(self.n_reservoirs):
            agent_id = f"reservoir_{i}"
            # 水库：个体安全势能 + 全局安全势能
            potentials[agent_id] = reservoir_safety_potentials[i] * 0.7 + global_safety_potential * 0.3

        for i in range(self.n_plants):
            agent_id = f"plant_{i}"
            # 水厂：主要关注全局安全
            potentials[agent_id] = global_safety_potential

        return potentials

    def _calculate_ecological_potential(self, state: Dict) -> Dict:
        """计算生态势函数"""
        potentials = {}

        target_flows = state.get('target_ecological_flows', np.zeros(self.n_reservoirs))
        actual_releases = state.get('ecological_releases', np.zeros(self.n_reservoirs))

        ecological_potentials = []
        for i in range(self.n_reservoirs):
            if target_flows[i] > 0:
                # 生态流量满足率
                eco_satisfaction = min(actual_releases[i] / target_flows[i], 2.0)  # 允许超额

                # 生态势函数：在0.8-1.2范围内为正
                if 0.8 <= eco_satisfaction <= 1.2:
                    eco_potential = 1.0 - abs(eco_satisfaction - 1.0) / 0.2
                else:
                    # 超出合理范围的惩罚
                    if eco_satisfaction < 0.8:
                        deficit = (0.8 - eco_satisfaction) / 0.8
                    else:  # eco_satisfaction > 1.2
                        deficit = (eco_satisfaction - 1.2) / 0.8
                    eco_potential = -deficit ** 2

                ecological_potentials.append(eco_potential)
            else:
                ecological_potentials.append(0.0)

        global_eco_potential = np.mean(ecological_potentials) if ecological_potentials else 0.0

        # 只有水库承担生态责任
        for i in range(self.n_reservoirs):
            agent_id = f"reservoir_{i}"
            potentials[agent_id] = ecological_potentials[i] * 0.6 + global_eco_potential * 0.4

        for i in range(self.n_plants):
            agent_id = f"plant_{i}"
            potentials[agent_id] = global_eco_potential * 0.2  # 水厂较小的生态责任

        return potentials

    def _get_all_agent_ids(self) -> List[str]:
        """获取所有智能体ID"""
        agent_ids = []
        for i in range(self.n_plants):
            agent_ids.append(f"plant_{i}")
        for i in range(self.n_reservoirs):
            agent_ids.append(f"reservoir_{i}")
        return agent_ids


class StabilizedMultiAgentRewardSystem:
    """稳定化的多智能体奖励系统"""

    def __init__(self, n_reservoirs=12, n_plants=3, max_episode_steps=168, distance_matrix=None, cost_factor=0.01):
        self.n_reservoirs = n_reservoirs
        self.n_plants = n_plants
        self.max_episode_steps = max_episode_steps
        
        # 🆕 成本优化参数
        self.distance_matrix = distance_matrix
        self.cost_factor = cost_factor  # 单位成本系数 p
        
        if distance_matrix is not None:
            print(f" 成本优化，距离矩阵形状: {distance_matrix.shape}")
            print(f"   成本系数: {cost_factor}")
        else:
            print(" 距离矩阵未提供，成本优化功能禁用")

        # 🔧 关键改进：简化和稳定的奖励尺度
        self.core_reward_weights = {
            'supply_satisfaction': 2.0,
            'reservoir_safety': 2.0,
            'ecological_compliance': 1.0,
            'stability_bonus': 2.0,  # 稳定性奖励
            'exploration_bonus': 0.0,  # 探索奖励权重
            'cost_optimization': 1.5   # 成本优化权重
        }

        # 关键添加：探索记录器
        self.exploration_recorder = ExplorationRecorder(n_reservoirs, n_plants, max_episode_steps)

        # 势函数奖励塑形器
        self.potential_shaper = PotentialBasedRewardShaper(n_reservoirs, n_plants)

        # 历史记录用于稳定性计算
        self.performance_history = deque(maxlen=50)
        self.reward_variance_history = deque(maxlen=20)

        # 奖励稳定化机制
        self.reward_stabilizer = RewardStabilizer(window_size=10)

        # 自适应权重调整
        self.adaptive_weights = AdaptiveWeightManager(self.core_reward_weights)

        print(" 稳定化多智能体奖励系统已初始化")
        # print(f"   - 势函数奖励塑形: 已启用")
        # print(f"   - 自适应权重调整: 已启用")
        # print(f"   - 奖励稳定化: 已启用")
        # print(f"   - 探索机制: 已启用")

    def calculate_rewards(self, state: Dict, actions: Dict, current_step: int,
                          episode_progress: float = None, is_terminal: bool = False,
                          prev_state: Dict = None) -> Tuple[Dict, Dict]:
        """主要奖励计算接口 """

        try:
            # Episode开始时记录
            if current_step == 0:
                self.exploration_recorder.record_episode_start()

            # 提取节水相关动作信息
            conservation_actions = self._extract_conservation_actions(actions, state)

            # 1. 计算核心奖励组件
            supply_rewards = self._calculate_supply_rewards(state)
            safety_rewards = self._calculate_safety_rewards(state, conservation_actions)
            ecological_rewards = self._calculate_ecological_rewards(state)

            # 2. 关键改进：势函数奖励塑形
            shaping_rewards = self.potential_shaper.calculate_potential_shaping(state, prev_state)

            # 3. 计算稳定性奖励
            stability_rewards = self._calculate_stability_rewards()

            # 4. 计算探索奖励
            exploration_rewards = self.exploration_recorder.record_actions(actions, state)

            # 5. 计算成本奖励
            cost_rewards = self._calculate_cost_rewards(state)

            # 5. 关键改进：自适应权重调整
            adapted_weights = self.adaptive_weights.get_adapted_weights(
                supply_performance=self._get_supply_performance(state),
                safety_performance=self._get_safety_performance(state),
                ecological_performance=self._get_ecological_performance(state)
            )

            # 6. 合并奖励 - 使用自适应权重
            final_rewards = {}
            for agent_id in self._get_all_agent_ids():
                core_reward = (
                        supply_rewards.get(agent_id, 0.0) * adapted_weights['supply_satisfaction'] +
                        safety_rewards.get(agent_id, 0.0) * adapted_weights['reservoir_safety'] +
                        ecological_rewards.get(agent_id, 0.0) * adapted_weights['ecological_compliance'] +
                        stability_rewards.get(agent_id, 0.0) * adapted_weights['stability_bonus'] +
                        exploration_rewards.get(agent_id, 0.0) * adapted_weights['exploration_bonus'] +
                        cost_rewards.get(agent_id, 0.0) * adapted_weights.get('cost_optimization', 1.5)
                )

                # 添加势函数塑形
                shaped_reward = core_reward + shaping_rewards.get(agent_id, 0.0)

                # 奖励裁剪机制
                clipped_reward = self._apply_reward_clipping(shaped_reward, agent_id)

                # 奖励稳定化
                stabilized_reward = self.reward_stabilizer.stabilize_reward(agent_id, clipped_reward)

                final_rewards[agent_id] = stabilized_reward

            # 全局奖励平衡
            final_rewards = self._apply_global_reward_balance(final_rewards)

            # 7.更新探索效果
            self.exploration_recorder.update_exploration_effectiveness(final_rewards, state)

            # 8. 更新历史记录
            self._update_performance_history(final_rewards, state)

            # 9. 构建信息字典
            info_dict = self._build_enhanced_info_dict(
                supply_rewards, safety_rewards, ecological_rewards,
                stability_rewards, shaping_rewards, exploration_rewards,
                adapted_weights, state
            )

            return final_rewards, info_dict

        except Exception as e:
            print(f"奖励计算错误: {e}")
            return self._get_default_rewards_and_infos()

    def _extract_conservation_actions(self, actions: Dict, state: Dict) -> List[Dict]:
        """提取节水相关的动作信息"""
        conservation_actions = []

        for i in range(self.n_reservoirs):
            reservoir_agent = f"reservoir_{i}"
            plant_agent = f"plant_{i}" if i < self.n_plants else None

            conservation_info = {
                'supply_reduction': 0.0,
                'demand_reduction': 0.0
            }

            # 从水库动作中提取供水减少信息
            if reservoir_agent in actions:
                reservoir_action = actions[reservoir_agent]
                release_ratio = reservoir_action.get('total_release_ratio', [0.1])[0]

                # 如果释放比例较低，可能是在节水
                if release_ratio < 0.05:  # 阈值需要调整
                    available_water = max(0, state['reservoirs'][i] - state['dead_capacity'][i])
                    potential_supply = available_water * 0.1  # 正常释放量
                    actual_supply = available_water * release_ratio
                    conservation_info['supply_reduction'] = potential_supply - actual_supply

            # 从水厂动作中提取需求减少信息
            if plant_agent and plant_agent in actions:
                plant_action = actions[plant_agent]
                demand_adjustment = plant_action.get('demand_adjustment', [1.0])[0]

                # 如果需求调整小于1，说明主动降低需求
                if demand_adjustment < 1.0:
                    base_demand = state['hourly_demand'][i] if i < len(state['hourly_demand']) else 0
                    conservation_info['demand_reduction'] = base_demand * (1.0 - demand_adjustment)

            conservation_actions.append(conservation_info)

        return conservation_actions

    # 新增辅助方法
    def _apply_reward_clipping(self, reward: float, agent_id: str) -> float:
        """应用奖励裁剪机制"""
        # 基础裁剪：防止极端值
        base_clipped = np.clip(reward, -50.0, 50.0)

        # 自适应裁剪：根据历史表现调整
        if hasattr(self, 'agent_reward_ranges'):
            if agent_id in self.agent_reward_ranges:
                min_reward, max_reward = self.agent_reward_ranges[agent_id]
                # 动态调整裁剪范围
                range_factor = 1.5
                adaptive_min = min_reward * range_factor
                adaptive_max = max_reward * range_factor
                adaptive_clipped = np.clip(base_clipped, adaptive_min, adaptive_max)
                return adaptive_clipped

        return base_clipped

    def _apply_global_reward_balance(self, rewards: Dict) -> Dict:
        """应用全局奖励平衡"""
        reward_values = list(rewards.values())

        # 检查奖励分布
        mean_reward = np.mean(reward_values)
        std_reward = np.std(reward_values)

        # 如果方差过大，进行平衡
        if std_reward > 20.0:  # 标准差阈值
            balanced_rewards = {}
            for agent_id, reward in rewards.items():
                # 向均值回归
                balanced_reward = mean_reward + 0.7 * (reward - mean_reward)
                balanced_rewards[agent_id] = balanced_reward
            return balanced_rewards

        return rewards

    def _update_agent_reward_ranges(self, rewards: Dict):
        """更新智能体奖励范围统计"""
        if not hasattr(self, 'agent_reward_ranges'):
            self.agent_reward_ranges = {}

        for agent_id, reward in rewards.items():
            if agent_id not in self.agent_reward_ranges:
                self.agent_reward_ranges[agent_id] = [reward, reward]
            else:
                min_reward, max_reward = self.agent_reward_ranges[agent_id]
                # 使用指数移动平均更新范围
                alpha = 0.05
                new_min = min(min_reward, reward) * alpha + min_reward * (1 - alpha)
                new_max = max(max_reward, reward) * alpha + max_reward * (1 - alpha)
                self.agent_reward_ranges[agent_id] = [new_min, new_max]


    def _calculate_supply_rewards(self, state: Dict) -> Dict:
        """增加节约用水奖励机制"""
        rewards = {}

        # 全局供水性能
        total_supply = np.sum(state['actual_supply'])
        total_demand = np.sum(state['hourly_demand'])
        global_satisfaction = min(total_supply / (total_demand + 1e-8), 1.0)

        # 水资源紧缺预警机制
        reservoir_levels = state['reservoirs'] / state['max_reservoir']
        avg_reservoir_level = np.mean(reservoir_levels)
        min_reservoir_level = np.min(reservoir_levels)

        # 节约用水激励因子
        if avg_reservoir_level < 0.3:  # 严重缺水
            conservation_factor = 0.3  # 期望供水满足率降至30%
            conservation_bonus = 3.0  # 高节约奖励
        elif avg_reservoir_level < 0.5:  # 中度缺水
            conservation_factor = 0.5 + 0.4 * (avg_reservoir_level - 0.3) / 0.2  # 30%-70%
            conservation_bonus = 2.0
        elif avg_reservoir_level < 0.7:  # 轻度缺水
            conservation_factor = 0.7 + 0.25 * (avg_reservoir_level - 0.5) / 0.2  # 70%-95%
            conservation_bonus = 1.0
        else:  # 水量充足
            conservation_factor = 0.95  # 正常供水
            conservation_bonus = 0.0


        # 根据水库水位调整供水奖励目标
        adaptive_target = conservation_factor

        def adaptive_supply_reward(satisfaction_rate, target_rate):
            """自适应供水奖励：根据水库水位调整最优供水率"""
            if satisfaction_rate <= target_rate:
                # 在目标范围内，越接近目标奖励越高
                return 2.0 * (satisfaction_rate / target_rate)
            else:
                # 超过目标时，根据水资源状况决定奖励
                if avg_reservoir_level > 0.7:
                    # 水量充足时，超额供水给予奖励
                    return 2.0 + 0.5 * (satisfaction_rate - target_rate)
                else:
                    # 水量不足时，超额供水给予惩罚
                    excess = satisfaction_rate - target_rate
                    return 2.0 - 1.0 * excess  # 惩罚超额供水

        base_reward = adaptive_supply_reward(global_satisfaction, adaptive_target)

        # 节约用水奖励
        if global_satisfaction <= adaptive_target and avg_reservoir_level < 0.5:
            conservation_reward = conservation_bonus * (
                        1.0 - abs(global_satisfaction - adaptive_target) / adaptive_target)
            base_reward += conservation_reward

        base_reward *= self.core_reward_weights['supply_satisfaction']

        # 智能体个体奖励分配
        total_agents = self.n_reservoirs + self.n_plants
        base_per_agent = base_reward / total_agents

        # 水厂：基于个体表现和节约意识的奖励分配
        for i in range(self.n_plants):
            agent_id = f"plant_{i}"
            individual_satisfaction = min(
                state['actual_supply'][i] / (state['hourly_demand'][i] + 1e-8), 1.0
            )

            # 个体节约奖励
            individual_conservation_reward = 0.0
            if individual_satisfaction <= adaptive_target and avg_reservoir_level < 0.5:
                individual_conservation_reward = conservation_bonus * 0.3

            rewards[agent_id] = base_per_agent + individual_conservation_reward

        # 水库：基于支持能力和保护意识的奖励分配
        for i in range(self.n_reservoirs):
            agent_id = f"reservoir_{i}"
            reservoir_level = reservoir_levels[i]

            # 水库保护奖励：水位较低时限制供水给予奖励
            prevention_reward = 0.0
            
            # 检查水位下降趋势
            level_trend_warning = 0.0
            if hasattr(self, 'reservoir_level_history') and len(self.reservoir_level_history) >= 5:
                recent_levels = [level[i] for level in list(self.reservoir_level_history)[-5:]]
                if len(recent_levels) >= 2:
                    trend = (recent_levels[-1] - recent_levels[0]) / len(recent_levels)
                    if trend < -0.02:  # 快速下降趋势
                        level_trend_warning = 1.0
                    elif trend < -0.01:  # 缓慢下降趋势
                        level_trend_warning = 0.5
            
            # 根据水位和趋势计算预防奖励
            if reservoir_level < 0.5:  # 水位较低时启动预防机制
                # 检查是否主动减少了供水（预防性节水）
                expected_supply = state['hourly_demand'][i] if i < len(state['hourly_demand']) else 0
                actual_supply = state['actual_supply'][i] if i < len(state['actual_supply']) else 0
                
                if actual_supply < expected_supply * 0.9:  # 减少了10%以上供水
                    # 基础预防奖励
                    supply_reduction_ratio = 1.0 - (actual_supply / (expected_supply + 1e-8))
                    base_prevention = 2.0 * supply_reduction_ratio
                    
                    # 水位越低，预防奖励越高
                    level_multiplier = (0.5 - reservoir_level) / 0.3  # 水位从50%降到20%时递增
                    level_multiplier = np.clip(level_multiplier, 0.0, 1.0)
                    
                    # 下降趋势预警加成
                    trend_multiplier = 1.0 + level_trend_warning * 0.5
                    
                    # 综合预防奖励
                    prevention_reward = base_prevention * (1.0 + level_multiplier) * trend_multiplier
                    
                    # 紧急情况下的额外奖励
                    if reservoir_level < 0.3 and supply_reduction_ratio > 0.3:
                        emergency_bonus = 3.0 * supply_reduction_ratio
                        prevention_reward += emergency_bonus
                        
                elif reservoir_level < 0.3 and actual_supply == 0:
                    # 极端情况：水位很低且完全停止供水
                    prevention_reward = 5.0 * (0.3 - reservoir_level) / 0.1

            # 原有的水库保护奖励（保持不变）
            protection_reward = 0.0
            if reservoir_level < 0.5:
                expected_supply = state['hourly_demand'][i] if i < len(state['hourly_demand']) else 0
                actual_supply = state['actual_supply'][i] if i < len(state['actual_supply']) else 0
                if actual_supply < expected_supply:
                    protection_reward = conservation_bonus * 0.4 * (1.0 - actual_supply / (expected_supply + 1e-8))

            rewards[agent_id] = base_per_agent + protection_reward + prevention_reward

        return rewards


    def _calculate_safety_rewards(self, state: Dict, conservation_actions: List[Dict] = None) -> Dict:
        """改进的安全奖励计算"""
        rewards = {}
        forced_spills = state.get('forced_spills', np.zeros(self.n_reservoirs))

        if conservation_actions is None:
            conservation_actions = []
            for i in range(self.n_reservoirs):
                conservation_actions.append({
                    'supply_reduction': 0.0,
                    'demand_reduction': 0.0,
                    'water_level': state['reservoirs'][i] / state['max_reservoir'][i]
                })

        def safety_reward_function(level_ratio, forced_spill_amount=0, conservation_info=None):
            """安全奖励函数 - 考虑泄洪和节水合理性"""
            if 0.2 <= level_ratio <= 0.75:
                return 1.0 * (1.0 - 2 * abs(level_ratio - 0.55))

            if level_ratio < 0.2:
                # 基础惩罚
                if level_ratio < 0.15:
                    base_penalty = -5 * (0.1 - level_ratio) / 0.1
                else:
                    base_penalty = -2 * (0.2 - level_ratio) / 0.1
                conservation_bonus = 0.0
                if conservation_info:
                    # 根据节水程度给予奖励
                    supply_reduction = conservation_info.get('supply_reduction', 0.0)
                    demand_reduction = conservation_info.get('demand_reduction', 0.0)
                    # 节水奖励计算
                    if supply_reduction > 0 or demand_reduction > 0:
                        conservation_bonus = min(2.0,
                                                 (supply_reduction + demand_reduction) / 1000)

                return base_penalty + conservation_bonus

            elif 0.75 < level_ratio <= 0.80:
                # 过渡区间：轻微惩罚，鼓励主动释放
                return -0.5 * (level_ratio - 0.75) / 0.1

            elif 0.80 < level_ratio <= 0.95:
                # 泄洪合理区间：如果主动泄洪，减少惩罚
                base_penalty = -1.0 * (level_ratio - 0.80) / 0.1
                spill_bonus = min(2.0, forced_spill_amount / 1000) if forced_spill_amount > 0 else 0.0
                return base_penalty + spill_bonus

            else:
                # 极危险区间：>95%，严厉惩罚
                base_penalty = -2.0 - 3.0 * (level_ratio - 0.95) / 0.05
                spill_bonus = min(1.0, forced_spill_amount / 2000) if forced_spill_amount > 0 else 0.0
                return base_penalty + spill_bonus

        # 计算每个水库的安全奖励
        total_safety_reward = 0
        for i in range(self.n_reservoirs):
            level_ratio = state['reservoirs'][i] / state['max_reservoir'][i]
            # 传递水库的泄洪量
            individual_forced_spill = forced_spills[i] if i < len(forced_spills) else 0
            conservation_info = conservation_actions[i]

            safety_score = safety_reward_function(
                level_ratio,
                individual_forced_spill,
                conservation_info
            )
            total_safety_reward += safety_score

        avg_safety_reward = total_safety_reward / self.n_reservoirs * self.core_reward_weights['reservoir_safety']

        # 分配给智能体
        for i in range(self.n_reservoirs):
            agent_id = f"reservoir_{i}"
            level_ratio = state['reservoirs'][i] / state['max_reservoir'][i]
            individual_forced_spill = forced_spills[i] if i < len(forced_spills) else 0
            individual_safety = safety_reward_function(level_ratio, individual_forced_spill) * self.core_reward_weights['reservoir_safety'] * 0.3
            rewards[agent_id] = avg_safety_reward * 0.7 + individual_safety

        for i in range(self.n_plants):
            agent_id = f"plant_{i}"
            rewards[agent_id] = avg_safety_reward * 0.3  # 水厂分担较少安全责任

        return rewards

    def _calculate_ecological_rewards(self, state: Dict) -> Dict:
        """生态奖励计算"""
        rewards = {}

        target_flows = state.get('target_ecological_flows', np.zeros(self.n_reservoirs))
        actual_releases = state.get('ecological_releases', np.zeros(self.n_reservoirs))

        if np.sum(target_flows) == 0:
            # 无生态流量要求
            for agent_id in self._get_all_agent_ids():
                rewards[agent_id] = 0.0
            return rewards

        ecological_scores = []
        for i in range(self.n_reservoirs):
            if target_flows[i] > 0:
                compliance_ratio = actual_releases[i] / target_flows[i]

                # 生态合规奖励函数
                if 0.85 <= compliance_ratio <= 1.15:
                    # 在合规范围内
                    eco_score = 1.0 - abs(compliance_ratio - 1.0) / 0.15
                else:
                    # 超出合规范围
                    if compliance_ratio < 0.85:
                        deficit = (0.85 - compliance_ratio) / 0.85
                    else:
                        deficit = (compliance_ratio - 1.15) / 0.15
                    eco_score = -0.3 * deficit

                ecological_scores.append(eco_score)

        if ecological_scores:
            avg_eco_score = np.mean(ecological_scores) * self.core_reward_weights['ecological_compliance']
        else:
            avg_eco_score = 0.0

        # 只有水库承担生态责任
        for i in range(self.n_reservoirs):
            agent_id = f"reservoir_{i}"
            rewards[agent_id] = avg_eco_score

        for i in range(self.n_plants):
            agent_id = f"plant_{i}"
            rewards[agent_id] = 0.0

        return rewards

    def _calculate_stability_rewards(self) -> Dict:
        """计算稳定性奖励"""
        rewards = {}

        if len(self.performance_history) < 10:
            # 历史数据不足
            for agent_id in self._get_all_agent_ids():
                rewards[agent_id] = 0.0
            return rewards

        # 计算性能稳定性
        recent_performance = list(self.performance_history)[-10:]
        performance_stability = 1.0 / (1.0 + np.std(recent_performance))

        # 稳定性奖励
        stability_reward = performance_stability * self.core_reward_weights['stability_bonus']

        # 平均分配给所有智能体
        per_agent_stability = stability_reward / (self.n_reservoirs + self.n_plants)

        for agent_id in self._get_all_agent_ids():
            rewards[agent_id] = per_agent_stability

        return rewards

    # 辅助方法
    def _get_supply_performance(self, state: Dict) -> float:
        total_supply = np.sum(state['actual_supply'])
        total_demand = np.sum(state['hourly_demand'])
        return min(total_supply / (total_demand + 1e-8), 1.0)

    def _get_safety_performance(self, state: Dict) -> float:
        reservoir_levels = state['reservoirs'] / state['max_reservoir']
        safe_count = np.sum((reservoir_levels >= 0.4) & (reservoir_levels <= 0.7))
        return safe_count / self.n_reservoirs

    def _get_ecological_performance(self, state: Dict) -> float:
        target_flows = state.get('target_ecological_flows', np.zeros(self.n_reservoirs))
        actual_releases = state.get('ecological_releases', np.zeros(self.n_reservoirs))

        if np.sum(target_flows) == 0:
            return 1.0

        compliance_scores = []
        for i in range(self.n_reservoirs):
            if target_flows[i] > 0:
                compliance = min(actual_releases[i] / target_flows[i], 1.0)
                compliance_scores.append(compliance)

        return np.mean(compliance_scores) if compliance_scores else 1.0

    def _update_performance_history(self, rewards: Dict, state: Dict):
        """更新性能历史"""
        avg_reward = np.mean(list(rewards.values()))
        self.performance_history.append(avg_reward)

        # 更新奖励方差历史
        reward_variance = np.var(list(rewards.values()))
        self.reward_variance_history.append(reward_variance)

    def _get_all_agent_ids(self) -> List[str]:
        """获取所有智能体ID"""
        agent_ids = []
        for i in range(self.n_plants):
            agent_ids.append(f"plant_{i}")
        for i in range(self.n_reservoirs):
            agent_ids.append(f"reservoir_{i}")
        return agent_ids

    def _build_enhanced_info_dict(self, supply_rewards, safety_rewards, ecological_rewards,
                                  stability_rewards, shaping_rewards, exploration_rewards,
                                  adapted_weights, state):
        """构建增强的信息字典"""
        
        # 添加探索信息
        exploration_summary = self.exploration_recorder.get_exploration_summary()
        agent_profiles = self.exploration_recorder.get_agent_exploration_profiles()
        
        # 奖励组件总结
        reward_components = {
            'supply': np.mean(list(supply_rewards.values())),
            'safety': np.mean(list(safety_rewards.values())),
            'ecological': np.mean(list(ecological_rewards.values())),
            'stability': np.mean(list(stability_rewards.values())),
            'shaping': np.mean(list(shaping_rewards.values())),
            'exploration': np.mean(list(exploration_rewards.values()))
        }

        # 权重信息
        weight_info = {
            'adapted_weights': adapted_weights,
            'weight_adjustments': {
                'supply_factor': adapted_weights['supply_satisfaction'] / self.core_reward_weights['supply_satisfaction'],
                'safety_factor': adapted_weights['reservoir_safety'] / self.core_reward_weights['reservoir_safety'],
                'ecological_factor': adapted_weights['ecological_compliance'] / self.core_reward_weights['ecological_compliance']
            }
        }

        # 性能指标
        performance_metrics = {
            'supply_performance': self._get_supply_performance(state),
            'safety_performance': self._get_safety_performance(state),
            'ecological_performance': self._get_ecological_performance(state)
        }

        # 构建每个智能体的信息
        agent_infos = {}
        for agent_id in self._get_all_agent_ids():
            agent_infos[agent_id] = {
                'total_reward': supply_rewards.get(agent_id, 0.0) + safety_rewards.get(agent_id, 0.0) + 
                               ecological_rewards.get(agent_id, 0.0) + stability_rewards.get(agent_id, 0.0) + 
                               shaping_rewards.get(agent_id, 0.0) + exploration_rewards.get(agent_id, 0.0),
                'supply_reward': supply_rewards.get(agent_id, 0.0),
                'safety_reward': safety_rewards.get(agent_id, 0.0),
                'ecological_reward': ecological_rewards.get(agent_id, 0.0),
                'stability_reward': stability_rewards.get(agent_id, 0.0),
                'shaping_reward': shaping_rewards.get(agent_id, 0.0),
                'exploration_reward': exploration_rewards.get(agent_id, 0.0)
            }

        # 全局探索信息
        global_info = {
            'phase': exploration_summary['current_state']['phase'],
            'reward_components': reward_components,
            'adapted_weights': weight_info,
            'performance_metrics': performance_metrics,
            'agent_rewards': agent_infos,
            'exploration_active': exploration_summary['current_state']['exploration_active'],
            'exploration_summary': exploration_summary,
            'agent_profiles': agent_profiles,
            'diversity_score': np.mean(list(self.exploration_recorder.action_diversity_scores.values()))
        }

        return global_info

    def _get_default_rewards_and_infos(self):
        """默认奖励和信息"""
        default_rewards = {agent_id: 0.0 for agent_id in self._get_all_agent_ids()}
        default_info = {
            'phase': 'error_recovery',
            'reward_components': {'supply': 0.0, 'safety': 0.0, 'ecological': 0.0, 'stability': 0.0, 'shaping': 0.0},
            'performance_metrics': {'supply_satisfaction_rate': 0.0, 'reservoir_safety_rate': 0.0,
                                    'ecological_compliance_rate': 0.0}
        }
        return default_rewards, default_info



    def _calculate_cost_rewards(self, state: Dict) -> Dict:
        """🆕 计算成本优化奖励 - 基于地理距离和供水量"""
        rewards = {}
        
        # 如果距离矩阵未提供，返回零奖励
        if self.distance_matrix is None:
            for agent_id in self._get_all_agent_ids():
                rewards[agent_id] = 0.0
            return rewards
        
        # 获取当前供水矩阵 - 需要从状态中重构
        supply_matrix = self._reconstruct_supply_matrix(state)
        
        # 计算总成本
        total_cost = 0.0
        for i in range(self.n_reservoirs):
            for j in range(self.n_plants):
                if i < supply_matrix.shape[0] and j < supply_matrix.shape[1]:
                    # 成本 = 单位成本系数 × 距离 × 供水量
                    cost = self.cost_factor * self.distance_matrix[i, j] * supply_matrix[i, j]
                    total_cost += cost
        
        # 🔧 成本奖励设计：成本越低，奖励越高
        if total_cost > 0:
            # 基础奖励计算：成本越低奖励越高
            max_possible_cost = np.sum(self.distance_matrix) * np.sum(state.get('actual_supply', [0])) * self.cost_factor
            if max_possible_cost > 0:
                cost_ratio = total_cost / max_possible_cost
                # 成本效率奖励：成本比例越低，奖励越高
                cost_efficiency_reward = 2.0 * (1.0 - cost_ratio)
            else:
                cost_efficiency_reward = 0.0
        else:
            cost_efficiency_reward = 0.0
        
        # 🎯 智能体奖励分配策略
        # 水库：根据其供水距离效率获得奖励
        for i in range(self.n_reservoirs):
            agent_id = f"reservoir_{i}"
            
            # 计算该水库的距离效率
            reservoir_supplies = supply_matrix[i, :] if i < supply_matrix.shape[0] else np.zeros(self.n_plants)
            reservoir_distances = self.distance_matrix[i, :] if i < self.distance_matrix.shape[0] else np.ones(self.n_plants) * 10.0
            
            if np.sum(reservoir_supplies) > 0:
                # 加权平均距离（按供水量加权）
                weighted_avg_distance = np.sum(reservoir_distances * reservoir_supplies) / np.sum(reservoir_supplies)
                
                # 距离效率：距离越短效率越高
                max_distance = np.max(self.distance_matrix) if self.distance_matrix.size > 0 else 10.0
                distance_efficiency = 1.0 - (weighted_avg_distance / max_distance)
                
                # 水库成本奖励
                reservoir_cost_reward = cost_efficiency_reward * distance_efficiency * 0.8
            else:
                reservoir_cost_reward = 0.0
            
            rewards[agent_id] = reservoir_cost_reward
        
        # 水厂：根据整体成本效率获得较小的奖励
        for i in range(self.n_plants):
            agent_id = f"plant_{i}"
            
            # 水厂获得整体成本效率的一小部分奖励
            plant_cost_reward = cost_efficiency_reward * 0.2 / self.n_plants
            rewards[agent_id] = plant_cost_reward
        
        return rewards
    
    def _reconstruct_supply_matrix(self, state: Dict) -> np.ndarray:
        """🔧 重构供水矩阵 - 从状态信息中估算各水库到各水厂的供水量"""
        supply_matrix = np.zeros((self.n_reservoirs, self.n_plants))
        
        actual_supply = state.get('actual_supply', np.zeros(self.n_plants))
        
        # 基于连接关系和水库容量比例分配
        for j in range(self.n_plants):
            plant_total_supply = actual_supply[j] if j < len(actual_supply) else 0.0
            
            if plant_total_supply > 0:
                # 找到连接到这个水厂的所有水库（基于固定连接矩阵）
                connected_reservoirs = list(range(self.n_reservoirs))  # 假设所有水库都连接到水厂
                
                if connected_reservoirs:
                    # 基于水库当前水量比例分配供水
                    reservoir_levels = []
                    for i in connected_reservoirs:
                        if i < len(state.get('reservoirs', [])):
                            level = state['reservoirs'][i]
                        else:
                            level = 0.0
                        reservoir_levels.append(max(0.0, level))
                    
                    total_available = sum(reservoir_levels)
                    
                    if total_available > 0:
                        # 按比例分配供水量
                        for k, i in enumerate(connected_reservoirs):
                            proportion = reservoir_levels[k] / total_available
                            supply_matrix[i, j] = plant_total_supply * proportion
                    else:
                        # 如果没有可用水量，平均分配（防止除零错误）
                        avg_supply = plant_total_supply / len(connected_reservoirs)
                        for i in connected_reservoirs:
                            supply_matrix[i, j] = avg_supply
        
        return supply_matrix


class RewardStabilizer:
    """奖励稳定化器 - 减少奖励方差"""

    def __init__(self, window_size=10, stabilization_factor=0.2, target_mean=0.0, target_std=2.0, max_clip=20.0):
        # 80%是原始奖励，20%是历史平均
        self.stabilization_factor = stabilization_factor
        
        # 增大标准差和裁剪范围，保持奖励差异性
        self.target_std = target_std
        self.max_clip = max_clip
        
        # 禁用过度标准化
        self.enable_normalization = False  # 新增开关
        
        self.agent_reward_history = defaultdict(lambda: deque(maxlen=window_size))
        
        # 全局奖励统计
        self.global_reward_stats = {
            'mean': 0.0,
            'std': 1.0,
            'history': deque(maxlen=1000)
        }

    def stabilize_reward(self, agent_id: str, raw_reward: float) -> float:
        """修复版：减少过度稳定化"""
        history = self.agent_reward_history[agent_id]
        history.append(raw_reward)

        if len(history) < 3:
            return raw_reward

        # 20%历史，80%原始
        moving_avg = np.mean(list(history))
        stabilized = (
                self.stabilization_factor * moving_avg +
                (1 - self.stabilization_factor) * raw_reward
        )

        # 条件标准化
        if self.enable_normalization:
            return self._normalize_reward(stabilized)
        else:
            return stabilized

    def _normalize_reward(self, reward: float) -> float:
        """标准化奖励到目标分布"""
        # Z-score标准化
        if self.global_reward_stats['std'] > 0:
            normalized = (reward - self.global_reward_stats['mean']) / self.global_reward_stats['std']
        else:
            normalized = reward
            
        # 调整到目标分布
        scaled = normalized * self.target_std + self.target_mean
        
        # 裁剪到合理范围
        clipped = np.clip(scaled, -self.max_clip, self.max_clip)
        
        return clipped

class AdaptiveWeightManager:
    """自适应权重管理器"""

    def __init__(self, base_weights: Dict[str, float]):
        self.base_weights = base_weights.copy()
        self.performance_history = {
            'supply': deque(maxlen=20),
            'safety': deque(maxlen=20),
            'ecological': deque(maxlen=20)
        }

    def get_adapted_weights(self, supply_performance: float,
                            safety_performance: float,
                            ecological_performance: float) -> Dict[str, float]:
        """获取自适应调整的权重"""

        # 更新性能历史
        self.performance_history['supply'].append(supply_performance)
        self.performance_history['safety'].append(safety_performance)
        self.performance_history['ecological'].append(ecological_performance)

        adapted_weights = self.base_weights.copy()

        if len(self.performance_history['supply']) < 5:
            return adapted_weights

        # 计算近期性能趋势
        recent_supply = np.mean(list(self.performance_history['supply'])[-5:])
        recent_safety = np.mean(list(self.performance_history['safety'])[-5:])
        recent_ecological = np.mean(list(self.performance_history['ecological'])[-5:])

        # 自适应调整：表现差的目标增加权重
        if recent_supply < 0.7:
            adapted_weights['supply_satisfaction'] *= 1.3
        elif recent_supply > 0.95:
            adapted_weights['supply_satisfaction'] *= 0.95

        if recent_safety < 0.6:
            adapted_weights['reservoir_safety'] *= 1.3
        elif recent_safety > 0.85:
            adapted_weights['reservoir_safety'] *= 0.95

        if recent_ecological < 0.8:
            adapted_weights['ecological_compliance'] *= 1.1

        challenge_bonus = 1.0
        if recent_supply > 0.9 and recent_safety > 0.8:
            # 当表现太好时，增加挑战性
            challenge_bonus = 1.2

        for key in adapted_weights:
            adapted_weights[key] *= challenge_bonus

        return adapted_weights
