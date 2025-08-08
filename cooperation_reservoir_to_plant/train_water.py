"""
水资源管理训练脚本
与 wandb agent 兼容
"""

import os
import sys
import torch
import numpy as np
import math
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import json
import wandb
from gym import spaces
from scipy.signal import savgol_filter
import seaborn as sns
import importlib.util
import time
import gc
from collections import deque
import threading
import argparse
import psutil


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

#  添加WandB优化设置
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['WANDB_INIT_TIMEOUT'] = '300'
os.environ['WANDB_SILENT'] = 'true'

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))


from training_config import get_config, get_optimized_hyperparameters, print_config_summary
CONFIG_AVAILABLE = True


# 导入wandb记录器
try:
    from wandb_data_extractor import EnhancedWandBLogger
    ENHANCED_LOGGER_AVAILABLE = True
    print(" WandB记录器已导入")
except ImportError:
    ENHANCED_LOGGER_AVAILABLE = False
    print(" WandB记录器不可用")


from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import update_linear_schedule


env_file_path = project_root / 'history' / 'MAPPO' / 'onpolicy' / 'envs' / 'water_env' / 'WaterManagementEnv.py'
spec = importlib.util.spec_from_file_location("WaterManagementEnv", env_file_path)
water_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(water_env_module)
WaterManagementEnv = water_env_module.WaterManagementEnv
print(f" 已从以下路径加载环境: {env_file_path}")

class SeparatedRewardLogger:
    """记录不同类型奖励的日志记录器"""

    def __init__(self, episode_length=96):
        self.episode_length = episode_length
        self.step_rewards = []
        self.episode_rewards = []
        self.component_rewards = {
            'supply': [],
            'safety': [],
            'ecological': [],
            'stability': [],
            'shaping': []
        }
        self.agent_rewards = {}

    def log_step_reward(self, step_reward, components=None, agent_rewards=None):
        """记录单步奖励"""
        self.step_rewards.append(step_reward)

        if components:
            for component, value in components.items():
                if component in self.component_rewards:
                    self.component_rewards[component].append(value)

        if agent_rewards:
            for agent_id, reward in agent_rewards.items():
                if agent_id not in self.agent_rewards:
                    self.agent_rewards[agent_id] = []
                self.agent_rewards[agent_id].append(reward)

    def log_episode_reward(self, episode_reward):
        """记录Episode奖励"""
        self.episode_rewards.append(episode_reward)

    def get_separated_metrics(self):
        """获取分离的指标"""
        if not self.step_rewards:
            return {}

        metrics = {
            # 基础指标
            'step_reward_mean': np.mean(self.step_rewards),
            'step_reward_std': np.std(self.step_rewards),
            'episode_reward_total': sum(self.step_rewards),
            'episode_reward_normalized': np.mean(self.step_rewards),
            'daily_reward': sum(self.step_rewards) / (len(self.step_rewards) / 24.0),
            'hourly_reward': np.mean(self.step_rewards),

            # 组件指标
            'components': {},
            'agents': {}
        }

        for component, rewards in self.component_rewards.items():
            if rewards:
                metrics['components'][component] = {
                    'mean': np.mean(rewards),
                    'total': sum(rewards),
                    'contribution': sum(rewards) / sum(self.step_rewards) if sum(self.step_rewards) != 0 else 0
                }

        # 智能体奖励
        for agent_id, rewards in self.agent_rewards.items():
            if rewards:
                metrics['agents'][agent_id] = {
                    'mean': np.mean(rewards),
                    'total': sum(rewards),
                    'std': np.std(rewards)
                }

        return metrics

    def reset(self):
        """重置记录器"""
        self.step_rewards = []
        self.component_rewards = {k: [] for k in self.component_rewards}
        self.agent_rewards = {}


def safe_tensor_to_numpy(tensor, device_type='cpu'):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        return np.array(tensor)


def safe_numpy_to_tensor(array, device, dtype=torch.float32):
    if isinstance(array, torch.Tensor):
        return array.to(device)
    else:
        return torch.from_numpy(np.array(array, dtype=dtype.numpy())).to(device)


def make_train_env(args):
    """环境创建函数"""
    try:
        env = WaterManagementEnv(
            num_reservoirs=args.num_reservoirs,
            num_plants=args.num_plants,
            use_fixed_connections=args.use_fixed_connections,
            continuous_management=args.continuous_management,
            max_episode_steps=args.episode_length,
            enable_optimizations=getattr(args, 'enable_optimizations', True)
        )

        if getattr(args, 'enable_optimizations', False):
            if hasattr(env, 'enable_training_optimizations'):
                env.enable_training_optimizations()
                print("动态启用训练优化功能")

            # 获取优化状态
            if hasattr(env, 'get_optimization_status'):
                status = env.get_optimization_status()
                print(f"   - 奖励系统: {status.get('reward_system_type', 'Unknown')}")
                print(f"   - 观测管理器: {status.get('obs_manager_type', 'Unknown')}")
                print(f"   - 观测维度: {status.get('obs_dimension', 'Unknown')}")

        print(f"训练环境创建成功: {len(env.agents)} 个智能体")

        if not hasattr(env, 'observation_space'):
            raise AttributeError("环境缺少observation_space属性")
        if not hasattr(env, 'agents'):
            raise AttributeError("环境缺少agents属性")

        return env

    except Exception as e:
        print(f"环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def convert_action_space_for_mappo(env):
    """动作空间的转换"""
    action_dims = []

    # 是否使用简化动作空间
    use_simplified = getattr(env, 'use_simplified_actions', False)

    if use_simplified:
        for agent_id in env.agents:
            action_dims.append(2)
        max_action_dim = 2
    else:
        for agent_id in env.agents:
            if agent_id.startswith('reservoir_'):
                action_dims.append(7)
            else:
                action_dims.append(3)
        max_action_dim = max(action_dims)

    print(f"动作空间统一：最大维度 {max_action_dim}")

    action_spaces = []
    for agent_id in env.agents:
        action_spaces.append(spaces.Box(low=0.0, high=1.0, shape=(max_action_dim,), dtype=np.float32))

    return action_spaces


def init_hierarchical_policy(env, args):
    """针对大规模系统的策略初始化"""
    obs_space = env.observation_space
    act_space_list = convert_action_space_for_mappo(env)

    first_agent = list(env.agents)[0]
    obs_dim = obs_space[first_agent].shape[0]
    num_agents = len(env.agents)

    print(f" 大规模策略初始化: {num_agents}个智能体，观察维度: {obs_dim}")

    # 大规模agent下的网络架构调整
    if num_agents >= 15:
        args.hidden_size = 128 if args.enable_optimizations else 256
        args.layer_N = 3
        args.entropy_coef = 0.02  # 增加探索
        args.lr = 3e-5 if args.enable_optimizations else 4e-5  # 降低学习率
        args.critic_lr = 1e-4
        print("大规模下优化配置")

    print(f"   - Hidden size: {args.hidden_size}")
    print(f"   - Layers: {args.layer_N}")
    print(f"   - Learning rates: actor={args.lr}, critic={args.critic_lr}")
    print(f"   - Entropy coef: {args.entropy_coef}")

    unified_obs_space = obs_space[first_agent]
    unified_action_space = act_space_list[0]

    device = args.device
    policy = R_MAPPOPolicy(args, unified_obs_space, unified_obs_space, unified_action_space, device=device)
    trainer = R_MAPPO(args, policy, device=device)

    print(f" 策略网络初始化完成: {num_agents}个智能体")
    return policy, trainer


def convert_actions_from_mappo(env, mappo_actions):
    """将MAPPO输出的动作转换为环境可接受的格式"""
    actions = {}
    num_plants = env.num_plants

    for i, agent_id in enumerate(env.agents):
        agent_action = mappo_actions[i]

        if agent_id.startswith('reservoir_'):
            # 提取水库ID
            res_id = int(agent_id.split('_')[1])

            # 动态计算分配权重的索引范围
            # 确保我们不会超出动作数组的范围
            allocation_weights = []
            connected_plants = []

            # 找出哪些水厂与该水库相连
            for p_idx in range(num_plants):
                if env.connections[res_id, p_idx]:
                    connected_plants.append(p_idx)

            # 如果有连接的水厂，计算权重
            if connected_plants:
                # 从动作中提取原始权重
                raw_weights = agent_action[1:1 + len(connected_plants)]
                # 归一化权重
                if np.sum(raw_weights) > 0:
                    allocation_weights = raw_weights / np.sum(raw_weights)
                else:
                    allocation_weights = np.ones(len(connected_plants)) / len(connected_plants)

            # 安全地获取紧急放水标志 - 使用最后一个元素而不是固定偏移
            emergency_index = min(len(agent_action) - 1, 1 + num_plants)
            emergency_release = int(np.round(np.clip(agent_action[emergency_index], 0, 1)))

            actions[agent_id] = {
                'total_release_ratio': [float(np.clip(agent_action[0], 0, 1))],
                'allocation_weights': allocation_weights.tolist(),
                'emergency_release': emergency_release
            }

        elif agent_id.startswith('plant_'):
            actions[agent_id] = {
                'demand_adjustment': [float(np.clip(agent_action[0], 0.5, 1.5))],
                'priority_level': int(np.clip(np.round(agent_action[1] * 2), 0, 2)),
                'storage_strategy': [float(np.clip(agent_action[2], 0, 1))]
            }

    return actions


class EnhancedTrainingMonitor:
    """用于训练监控"""

    def __init__(self, log_interval=10):
        self.log_interval = log_interval

        # 监控数据
        self.metrics = {
            'episode_rewards': deque(maxlen=1000),
            'episode_lengths': deque(maxlen=1000),
            'training_steps': deque(maxlen=1000),
            'eval_rewards': deque(maxlen=100),
            'system_metrics': deque(maxlen=1000)
        }
        self.start_time = time.time()

    def connect_environment(self, env):
        """仅用于兼容性，不执行任何操作"""
        pass

    def log_episode(self, episode, episode_reward, episode_steps, total_steps, infos=None):
        """记录episode信息"""
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['episode_lengths'].append(episode_steps)
        self.metrics['training_steps'].append(total_steps)

        try:
            import wandb
            if wandb.run is not None:
                # 立即记录到 wandb
                episode_time = time.time() - self.start_time
                log_data = {
                    "monitor/episode": int(episode),
                    "monitor/episode_reward": float(episode_reward),
                    "monitor/episode_steps": int(episode_steps),
                    "monitor/total_steps": int(total_steps),
                    "monitor/training_time": float(episode_time),
                    "global_step": int(total_steps)
                }

                # 添加详细信息
                if infos:
                    first_agent_info = list(infos.values())[0]
                    if isinstance(first_agent_info, dict):
                        if 'phase' in first_agent_info:
                            log_data["monitor/phase"] = str(first_agent_info['phase'])
                        if 'exploration_active' in first_agent_info:
                            log_data["monitor/exploration_active"] = bool(first_agent_info['exploration_active'])

                wandb.log(log_data)
                print(f" Monitor强制记录: Episode {episode}")
        except Exception as e:
            print(f" Monitor wandb 记录失败: {e}")


        if episode % self.log_interval == 0:
            avg_reward = np.mean(list(self.metrics['episode_rewards'])[-10:])
            avg_length = np.mean(list(self.metrics['episode_lengths'])[-10:])
            elapsed_time = time.time() - self.start_time

            print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                  f"AvgReward(10)={avg_reward:.2f}, Length={episode_steps}, "
                  f"Steps={total_steps}, Time={elapsed_time:.1f}s")

        try:
            import wandb
            if wandb.run is not None:
                monitor_log_data = {
                    "monitor/episode": int(episode),
                    "monitor/episode_reward": float(episode_reward),
                    "monitor/episode_steps": int(episode_steps),
                    "monitor/total_steps": int(total_steps),
                    "monitor/timestamp": time.time()
                }

                if infos:
                    for agent_id, info in infos.items():
                        if isinstance(info, dict):
                            if 'phase' in info:
                                monitor_log_data["monitor/current_phase"] = str(info['phase'])
                            break

                wandb.log(monitor_log_data)
                print(f"Monitor wandb 记录: Episode {episode}")
        except Exception as e:
            print(f" Monitor wandb 记录失败: {e}")

    def log_eval(self, eval_reward, eval_metrics=None):
        """记录评估信息"""
        self.metrics['eval_rewards'].append(eval_reward)
        print(f" 评估奖励: {eval_reward:.3f}")

    def log_enhanced_analysis(self, env):
        """分析信息"""
        try:
            if hasattr(env, 'get_enhanced_reward_analysis'):
                analysis = env.get_enhanced_reward_analysis()

                if 'status' not in analysis:
                    exploration_summary = analysis.get('exploration_summary', {})
                    current_state = exploration_summary.get('current_state', {})

                    phase = current_state.get('phase', 'unknown')
                    effectiveness = current_state.get('exploration_effectiveness', 0.0)

                    # 记录到系统指标
                    self.metrics['system_metrics'].append({
                        'timestamp': time.time(),
                        'phase': phase,
                        'effectiveness': effectiveness,
                        'analysis': analysis
                    })

                    print(f"增强分析: Phase={phase}, Effectiveness={effectiveness:.3f}")

                    return analysis
        except Exception as e:
            print(f" 增强分析记录失败: {e}")

        return None

    def get_stats(self):
        """获取统计信息"""
        if not self.metrics['episode_rewards']:
            return {}

        return {
            'avg_reward': np.mean(self.metrics['episode_rewards']),
            'std_reward': np.std(self.metrics['episode_rewards']),
            'max_reward': np.max(self.metrics['episode_rewards']),
            'avg_length': np.mean(self.metrics['episode_lengths']),
            'total_episodes': len(self.metrics['episode_rewards']),
        }

    def cleanup(self):
        """清理资源"""
        pass


def safe_wandb_log(data, step=None):
    """安全的W&B记录函数"""
    try:
        if wandb.run is not None:
            if step is not None:
                wandb.log(data, step=step)
            else:
                wandb.log(data)
            return True
    except Exception as e:
        print(f" W&B记录失败: {e}")
        return False


def initialize_wandb(args):
    """初始化wandb"""
    if args.use_wandb:
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                project_suffix = "_optimized" if getattr(args, 'enable_optimizations', True) else "_standard"
                run_name = f"water_management{project_suffix}_{timestamp}"

                tags = ["water_management", "mappo", "multi_agent"]
                if getattr(args, 'enable_optimizations', False):
                    tags.extend(["optimized", "simplified_reward", "enhanced_exploration", "reduced_obs"])

                # 配置
                config = {
                    'num_reservoirs': args.num_reservoirs,
                    'num_plants': args.num_plants,
                    'episode_length': args.episode_length,
                    'num_env_steps': args.num_env_steps,
                    'lr': args.lr,
                    'critic_lr': getattr(args, 'critic_lr', args.lr),
                    'hidden_size': args.hidden_size,
                    'ppo_epoch': args.ppo_epoch,
                    'entropy_coef': args.entropy_coef,
                    'clip_param': args.clip_param,
                    'enable_optimizations': getattr(args, 'enable_optimizations', False),
                    'continuous_management': args.continuous_management,
                    'algorithm_name': getattr(args, 'algorithm_name', 'mappo'),
                    'experiment_name': getattr(args, 'experiment_name', 'water_management'),
                    'seed': args.seed
                }

                project_name = getattr(args, 'wandb_project', 'uncategorized')

                print(f"尝试初始化 WandB (尝试 {retry_count + 1}/{max_retries})...")

                wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config,
                    group="optimized_training" if getattr(args, 'enable_optimizations', False) else "standard_training",
                    tags=tags,
                    settings=wandb.Settings(
                        init_timeout=300,
                        start_method='thread',
                        _disable_stats=True,
                        _disable_meta=True,
                        console='off',
                        # 数据优化设置
                        _disable_code=True,
                        _disable_machine_info=True,
                        save_code=False,
                        log_interval=100,
                        _sync_run_history=False,
                        _sync_plots=False,
                        console_logging=False,
                        run_mode="online",
                        _disable_git=True,
                        _save_requirements=False
                    )
                )
                print(f" WandB初始化成功: {run_name}")
                print(f"项目: {project_name}")
                print(f" 标签: {', '.join(tags)}")
                return True

            except Exception as e:
                retry_count += 1
                print(f" WandB初始化失败 (尝试 {retry_count}/{max_retries}): {e}")

                if retry_count < max_retries:
                    print(f"等待 {retry_count * 5} 秒后重试...")
                    time.sleep(retry_count * 5)
                else:
                    print("尝试失败，切换到离线模式")

                    try:
                        wandb.init(
                            project=project_name,
                            name=run_name,
                            config=config,
                            group="optimized_training" if getattr(args, 'enable_optimizations',
                                                                  False) else "standard_training",
                            tags=tags,
                            mode="offline"
                        )
                        print(f"WandB离线模式启动成功: {run_name}")
                        return True
                    except Exception as offline_e:
                        print(f"离线模式也失败: {offline_e}")
                        print(f" 建议: 1) 检查网络连接 2) 尝试 'wandb login --relogin' 3) 检查项目权限")
                        args.use_wandb = False
                        return False

        return False
    return False


def train(args):
    """增强版训练函数"""
    print("======================开始水资源管理训练==========================")

    print("训练参数检查:")
    print(f"   - use_wandb: {getattr(args, 'use_wandb', 'NOT_SET')}")
    print(f"   - wandb_project: {getattr(args, 'wandb_project', 'NOT_SET')}")
    print(f"   - run_dir: {getattr(args, 'run_dir', 'NOT_SET')}")
    print(f"   - num_env_steps: {getattr(args, 'num_env_steps', 'NOT_SET')}")
    print(f"   - episode_length: {getattr(args, 'episode_length', 'NOT_SET')}")
    print(f"   - ENHANCED_LOGGER_AVAILABLE: {ENHANCED_LOGGER_AVAILABLE}")

    if not hasattr(args, 'use_wandb'):
        args.use_wandb = True
        print("启用use_wandb")

    if not hasattr(args, 'wandb_project'):
        args.wandb_project = 'uncategorized'

    if not hasattr(args, 'enable_optimizations'):
        args.enable_optimizations = False

    if not hasattr(args, 'log_interval'):
        args.log_interval = 10

    if not hasattr(args, 'device'):
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化WandB记录器
    enhanced_logger = None

    if args.use_wandb and ENHANCED_LOGGER_AVAILABLE:
        try:
            enhanced_logger = EnhancedWandBLogger(log_dir=f"{args.run_dir}/wandb_data")
            print(f" WandB记录器初始化成功: {args.run_dir}/wandb_data")
        except Exception as e:
            print(f" WandB记录器初始化失败: {e}")
            enhanced_logger = None
    else:
        print(f" 不满足初始化条件: use_wandb={args.use_wandb}, ENHANCED_LOGGER_AVAILABLE={ENHANCED_LOGGER_AVAILABLE}")

    print(f" Enhanced logger 最终状态: {enhanced_logger is not None}")
    if enhanced_logger:
        print(f" Enhanced logger 保存目录: {enhanced_logger.log_dir}")

    # 初始化WandB
    if args.use_wandb and not hasattr(wandb, 'run') or wandb.run is None:
        use_wandb = initialize_wandb(args)
        print(f" WandB初始化结果: {use_wandb}")
    else:
        use_wandb = args.use_wandb
        if use_wandb:
            print(f"使用现有的 WandB 会话: {wandb.run.name if wandb.run else 'Unknown'}")

    device = args.device
    print(f" 使用设备: {device}")

    #  创建环境
    print(" 开始创建环境...")
    env = make_train_env(args)
    eval_env = make_train_env(args)
    print(" 环境创建完成")

    # 初始化策略
    print(" 开始初始化策略...")
    policy, trainer = init_hierarchical_policy(env, args)
    print(" 策略初始化完成")

    n_rollout_threads = 1

    # 计算动作空间
    act_space_list = convert_action_space_for_mappo(env)
    first_agent = list(env.agents)[0]
    unified_obs_space = env.observation_space[first_agent]
    unified_action_space = act_space_list[0]

    buffer = SharedReplayBuffer(
        args, args.num_agents, unified_obs_space,
        unified_obs_space, unified_action_space
    )
    print(" [DEBUG] Buffer创建完成")

    # 初始化监控器
    monitor = EnhancedTrainingMonitor(
        log_interval=args.log_interval,
    )

    print(" [DEBUG] 监控器创建完成")

    # 用于追踪最佳性能的变量
    best_eval_reward = -np.inf
    no_improvement_count = 0
    max_no_improvement = 50  # 50次评估无改善则早停

    # 用于追踪收敛的变量
    episode_rewards = deque(maxlen=100)  # 保存最近100个episode的奖励
    total_steps = 0
    episode = 0

    if getattr(args, 'enable_optimizations', False):
        print(" 优化已启用:")
    else:
        print(" 使用标准训练配置")

    #  提前计算最大步数
    max_episodes = args.num_env_steps // args.episode_length
    print(f" 目标: {max_episodes} episodes, 每个episode {args.episode_length} 步")

    #  训练状态标记
    training_completed = False

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f" 开始训练循环，最大episodes: {getattr(args, 'max_episodes', 1000)}")
    print(f" [DEBUG] 训练循环开始前状态:")
    print(f"   - enhanced_logger: {enhanced_logger is not None}")
    print(f"   - use_wandb: {use_wandb}")
    print(f"   - training_completed: {training_completed}")
    print(f"   - max_episodes: {max_episodes}")

    # 训练主循环
    try:
        while not training_completed:
            if total_steps >= args.num_env_steps:
                print(f" 达到最大训练步数 {args.num_env_steps:,}")
                break

            if episode >= getattr(args, 'max_episodes', 1000):
                print(f" 达到最大训练回合数 {getattr(args, 'max_episodes', 1000)}")
                break

            # Episode开始
            episode_start_time = time.time()

            # [DEBUG] 每10个episode打印一次状态
            if episode % 10 == 0:
                print(f" [DEBUG] Episode {episode + 1} 开始:")
                print(f"   - enhanced_logger存在: {enhanced_logger is not None}")
                print(f"   - total_steps: {total_steps}")
                print(f"   - 历史数据长度: {len(enhanced_logger.episode_history) if enhanced_logger else 0}")

            try:
                # 重置环境
                obs = env.reset()

                rnn_states_actor = np.zeros((args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
                rnn_states_critic = np.zeros_like(rnn_states_actor)
                masks = np.ones((args.num_agents, 1), dtype=np.float32)

                episode_reward = 0
                episode_steps = 0
                episode_done = False

                # Episode循环
                while not episode_done and episode_steps < args.episode_length:
                    if total_steps >= args.num_env_steps:
                        training_completed = True
                        break

                    try:
                        # 环境交互
                        obs_list = [obs[agent] for agent in env.agents]

                        with torch.no_grad():
                            obs_array = np.array(obs_list, dtype=np.float32)
                            obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(args.device)
                            share_obs_tensor = obs_tensor

                            rnn_states_actor_tensor = torch.from_numpy(rnn_states_actor).unsqueeze(0).to(args.device)
                            rnn_states_critic_tensor = torch.from_numpy(rnn_states_critic).unsqueeze(0).to(args.device)
                            masks_tensor = torch.from_numpy(masks).unsqueeze(0).to(args.device)

                            # 获取动作
                            value, action, action_log_prob, rnn_states_actor_new, rnn_states_critic_new = policy.get_actions(
                                share_obs_tensor,
                                obs_tensor,
                                rnn_states_actor_tensor,
                                rnn_states_critic_tensor,
                                masks_tensor
                            )

                        rnn_states_actor = safe_tensor_to_numpy(rnn_states_actor_new.squeeze(0))
                        rnn_states_critic = safe_tensor_to_numpy(rnn_states_critic_new.squeeze(0))

                        actions_np = safe_tensor_to_numpy(action.squeeze(0))
                        actions_dict = convert_actions_from_mappo(env, actions_np)

                        # 环境步进
                        next_obs, rewards, dones, truncations, infos = env.step(actions_dict)

                        # 更新masks
                        masks_old = masks.copy()
                        masks = np.ones((args.num_agents, 1), dtype=np.float32)
                        for i, agent_id in enumerate(env.agents):
                            if dones[agent_id] or any(t for t in truncations.values() if t):
                                masks[i] = 0.0

                        rewards_list = [rewards[agent] for agent in env.agents]
                        rewards_np = np.array(rewards_list, dtype=np.float32).reshape(1, -1, 1)

                        # 转换numpy状态为tensor用于buffer存储
                        rnn_states_actor_for_buffer = torch.from_numpy(rnn_states_actor).to(args.device)
                        rnn_states_critic_for_buffer = torch.from_numpy(rnn_states_critic).to(args.device)
                        masks_old_tensor = torch.from_numpy(masks_old).to(args.device)

                        buffer.insert(
                            share_obs_tensor,
                            obs_tensor,
                            rnn_states_actor_for_buffer,
                            rnn_states_critic_for_buffer,
                            action,
                            action_log_prob,
                            value,
                            torch.from_numpy(rewards_np).to(args.device),
                            masks_old_tensor
                        )

                        # 更新状态
                        obs = next_obs
                        step_reward = np.mean(list(rewards.values()))
                        episode_reward += step_reward / args.episode_length
                        total_steps += 1
                        episode_steps += 1

                        # 添加step级别的详细wandb记录
                        if use_wandb and total_steps % 100 == 0:  # 每100步记录一次
                            try:
                                step_log_data = {
                                    "step/total_steps": int(total_steps),
                                    "step/episode": int(episode + 1),
                                    "step/episode_step": int(episode_steps),
                                    "step/reward": float(step_reward),
                                    "step/episode_reward_cumulative": float(episode_reward),
                                    "step/timestamp": time.time()
                                }

                                # 记录平均奖励
                                if rewards:
                                    agent_rewards = list(rewards.values())
                                    step_log_data["step/agent_rewards_mean"] = float(np.mean(agent_rewards))
                                    step_log_data["step/agent_rewards_std"] = float(np.std(agent_rewards))
                                    step_log_data["step/agent_rewards_max"] = float(np.max(agent_rewards))
                                    step_log_data["step/agent_rewards_min"] = float(np.min(agent_rewards))

                                if 'value' in locals():
                                    step_log_data["step/value_function"] = float(value.mean().item())

                                if 'actions_np' in locals():
                                    step_log_data["step/action_mean"] = float(np.mean(actions_np))
                                    step_log_data["step/action_std"] = float(np.std(actions_np))

                                if total_steps % 500 == 0:  # 每500步记录一次系统指标
                                    step_log_data["system/memory_usage_mb"] = float(psutil.virtual_memory().used / 1024 / 1024)
                                    step_log_data["system/memory_percent"] = float(psutil.virtual_memory().percent)
                                    step_log_data["system/cpu_percent"] = float(psutil.cpu_percent())

                                    # 添加CUDA内存监控
                                    if torch.cuda.is_available():
                                        step_log_data["system/gpu_memory_used_mb"] = float(torch.cuda.memory_allocated() / 1024 / 1024)
                                        step_log_data["system/gpu_memory_cached_mb"] = float(torch.cuda.memory_reserved() / 1024 / 1024)

                                # 记录环境状态信息
                                if infos:
                                    first_agent_info = list(infos.values())[0]
                                    if isinstance(first_agent_info, dict):
                                        if 'supply_satisfaction_rate' in first_agent_info:
                                            step_log_data["step/supply_satisfaction"] = float(first_agent_info['supply_satisfaction_rate'])
                                        if 'avg_reservoir_level' in first_agent_info:
                                            step_log_data["step/reservoir_level"] = float(first_agent_info['avg_reservoir_level'])
                                        if 'ecological_flow_deviation' in first_agent_info:
                                            step_log_data["step/ecological_deviation"] = float(first_agent_info['ecological_flow_deviation'])

                                        # 奖励记录
                                        if 'reward_components' in first_agent_info:
                                            reward_components = first_agent_info['reward_components']
                                            if isinstance(reward_components, dict):

                                                total_reward = sum(v for v in reward_components.values() if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))
                                                step_log_data["step/reward_components_total"] = float(total_reward)

                                        if 'phase' in first_agent_info:
                                            step_log_data["step/phase"] = str(first_agent_info['phase'])
                                        if 'exploration_active' in first_agent_info:
                                            step_log_data["step/exploration_active"] = int(bool(first_agent_info['exploration_active']))

                                # 水库指标
                                if hasattr(env, 'reservoirs') and hasattr(env, 'max_reservoir'):
                                    total_volume = np.sum(env.reservoirs)
                                    total_capacity = np.sum(env.max_reservoir)
                                    reservoir_levels = env.reservoirs / env.max_reservoir

                                    step_log_data["reservoirs/total_volume"] = float(total_volume)
                                    step_log_data["reservoirs/total_capacity"] = float(total_capacity)
                                    step_log_data["reservoirs/avg_level"] = float(np.mean(reservoir_levels))
                                    step_log_data["reservoirs/min_level"] = float(np.min(reservoir_levels))
                                    step_log_data["reservoirs/max_level"] = float(np.max(reservoir_levels))
                                    step_log_data["reservoirs/level_std"] = float(np.std(reservoir_levels))

                                # 水厂指标
                                if hasattr(env, 'plants_demand'):
                                    total_demand = np.sum(env.plants_demand)
                                    step_log_data["plants/total_demand"] = float(total_demand)
                                    step_log_data["plants/avg_demand"] = float(np.mean(env.plants_demand))

                                # 供水指标
                                if hasattr(env, '_last_actual_supply'):
                                    actual_supply = env._last_actual_supply
                                    total_supply = np.sum(actual_supply)
                                    step_log_data["supply/total_supply"] = float(total_supply)
                                    step_log_data["supply/avg_supply"] = float(np.mean(actual_supply))

                                    # 供需平衡
                                    if hasattr(env, 'plants_demand'):
                                        total_demand = np.sum(env.plants_demand) / 24.0  # 小时需求
                                        if total_demand > 0:
                                            step_log_data["balance/supply_demand_ratio"] = float(total_supply / total_demand)
                                            step_log_data["balance/demand_satisfaction_rate"] = float(min(1.0, total_supply / total_demand))

                                # 安全指标
                                if hasattr(env, 'reservoirs') and hasattr(env, 'max_reservoir'):
                                    reservoir_levels = env.reservoirs / env.max_reservoir
                                    low_level_count = np.sum(reservoir_levels < 0.4)
                                    critical_count = np.sum(reservoir_levels < 0.2)
                                    high_level_count = np.sum(reservoir_levels > 0.8)

                                    step_log_data["safety/low_level_reservoirs"] = int(low_level_count)
                                    step_log_data["safety/critical_level_reservoirs"] = int(critical_count)
                                    step_log_data["safety/high_level_reservoirs"] = int(high_level_count)
                                    step_log_data["safety/safe_reservoirs"] = int(len(env.reservoirs) - low_level_count - high_level_count - critical_count)

                                wandb.log(step_log_data, step=total_steps)

                            except Exception as e:
                                print(f" Step级别wandb记录失败: {e}")

                        if use_wandb and total_steps % 1000 == 0:
                            try:
                                detailed_snapshot = {
                                    "detailed/total_steps": int(total_steps),
                                    "detailed/episode": int(episode + 1),
                                    "detailed/timestamp": time.time()
                                }

                                # 详细的水库状态（每1000步记录一次）
                                if hasattr(env, 'reservoirs') and hasattr(env, 'max_reservoir'):
                                    for i in range(min(len(env.reservoirs), 5)):  # 只记录前5个水库
                                        current_level = float(env.reservoirs[i]) / float(env.max_reservoir[i]) if env.max_reservoir[i] > 0 else 0.0
                                        detailed_snapshot[f"detailed/reservoir_{i}_level"] = current_level

                                # 详细的智能体奖励（抽样记录）
                                if rewards:
                                    sample_agents = list(rewards.keys())[:5]
                                    for agent_id in sample_agents:
                                        detailed_snapshot[f"detailed/agent_rewards/{agent_id}"] = float(rewards[agent_id])

                                wandb.log(detailed_snapshot, step=total_steps)
                                print(f" 详细快照已记录: Step {total_steps}")

                            except Exception as e:
                                print(f" 详细快照记录失败: {e}")

                        # 检查episode结束
                        if all(dones.values()) or any(truncations.values()):
                            episode_done = True

                            # 计算优势和更新策略
                            try:
                                with torch.no_grad():
                                    next_obs_array = np.array([next_obs[agent] for agent in env.agents],
                                                              dtype=np.float32)
                                    next_obs_tensor = torch.from_numpy(next_obs_array).unsqueeze(0).to(args.device)

                                    final_rnn_states_critic = torch.from_numpy(rnn_states_critic).unsqueeze(0).to(
                                        args.device)
                                    final_masks = torch.from_numpy(masks).unsqueeze(0).to(args.device)

                                    next_value = policy.get_values(
                                        next_obs_tensor,
                                        final_rnn_states_critic,
                                        final_masks
                                    )

                                buffer.compute_returns(next_value, policy.critic)
                                train_info = trainer.train(buffer)
                                buffer.after_update()

                                # 更新的详细wandb记录
                                if use_wandb and train_info:
                                    try:
                                        policy_log_data = {
                                            "policy/episode": int(episode + 1),
                                            "policy/total_steps": int(total_steps),
                                            "policy/update_timestamp": time.time()
                                        }

                                        # 训练指标
                                        if 'value_loss' in train_info:
                                            policy_log_data["policy/value_loss"] = float(train_info['value_loss'])
                                        if 'policy_loss' in train_info:
                                            policy_log_data["policy/policy_loss"] = float(train_info['policy_loss'])
                                        if 'entropy' in train_info:
                                            policy_log_data["policy/entropy"] = float(train_info['entropy'])
                                        if 'ratio' in train_info:
                                            policy_log_data["policy/ratio"] = float(train_info['ratio'])
                                        if 'surr1' in train_info:
                                            policy_log_data["policy/surr1"] = float(train_info['surr1'])
                                        if 'surr2' in train_info:
                                            policy_log_data["policy/surr2"] = float(train_info['surr2'])
                                        if 'kl_div' in train_info:
                                            policy_log_data["policy/kl_divergence"] = float(train_info['kl_div'])
                                        if 'clip_frac' in train_info:
                                            policy_log_data["policy/clip_fraction"] = float(train_info['clip_frac'])
                                        if 'grad_norm' in train_info:
                                            policy_log_data["policy/grad_norm"] = float(train_info['grad_norm'])

                                        # 学习率
                                        if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
                                            policy_log_data["policy/learning_rate"] = float(trainer.lr_scheduler.get_last_lr()[0])
                                        elif hasattr(trainer, 'optimizer') and trainer.optimizer:
                                            policy_log_data["policy/learning_rate"] = float(trainer.optimizer.param_groups[0]['lr'])

                                        # 网络参数统计
                                        if hasattr(policy, 'actor') and policy.actor:
                                            actor_params = []
                                            for param in policy.actor.parameters():
                                                if param.requires_grad:
                                                    actor_params.append(param.data.cpu().numpy().flatten())
                                            if actor_params:
                                                actor_params = np.concatenate(actor_params)
                                                policy_log_data["policy/actor_param_mean"] = float(np.mean(actor_params))
                                                policy_log_data["policy/actor_param_std"] = float(np.std(actor_params))

                                        if hasattr(policy, 'critic') and policy.critic:
                                            critic_params = []
                                            for param in policy.critic.parameters():
                                                if param.requires_grad:
                                                    critic_params.append(param.data.cpu().numpy().flatten())
                                            if critic_params:
                                                critic_params = np.concatenate(critic_params)
                                                policy_log_data["policy/critic_param_mean"] = float(np.mean(critic_params))
                                                policy_log_data["policy/critic_param_std"] = float(np.std(critic_params))

                                        wandb.log(policy_log_data, step=total_steps)
                                        print(f" 策略更新记录: Episode {episode + 1}, 指标数: {len(policy_log_data)}")

                                    except Exception as e:
                                        print(f" 策略更新wandb记录失败: {e}")

                            except Exception as e:
                                print(f" 策略更新出错: {e}")
                                import traceback
                                traceback.print_exc()
                                episode_done = True

                            break

                    except Exception as e:
                        print(f" 环境交互出错: {e}")
                        import traceback
                        traceback.print_exc()

                        infos = {agent: {} for agent in env.agents}
                        episode_done = True
                        break

                if training_completed:
                    break

                episode_time = time.time() - episode_start_time
                if 'infos' not in locals():
                    infos = {agent: {} for agent in env.agents}

                # 标准监控器记录
                monitor.log_episode(episode + 1, episode_reward, episode_steps, total_steps, infos)

                # 增强WandB记录器记录（确保完整历史数据保存）
                if enhanced_logger:
                    try:
                        print(f" [DEBUG] Episode {episode + 1} 尝试记录到增强记录器...")
                        enhanced_logger.log_episode_complete(
                            episode + 1, episode_reward, episode_steps, total_steps, infos
                        )
                        print(f" [DEBUG] 增强记录器已保存Episode {episode + 1}完整数据，当前历史长度: {len(enhanced_logger.episode_history)}")
                    except Exception as e:
                        print(f"️ 增强记录器记录失败: {e}")
                else:
                    print(f"️ [DEBUG] Episode {episode + 1} 跳过增强记录器记录（enhanced_logger为None）")

            except Exception as e:
                print(f" Episode {episode + 1} 出错: {e}")
                import traceback
                traceback.print_exc()

                if 'infos' not in locals():
                    infos = {agent: {} for agent in env.agents}

                episode_time = time.time() - episode_start_time
                monitor.log_episode(episode + 1, 0.0, episode_steps, total_steps, infos)

                if enhanced_logger:
                    try:
                        enhanced_logger.log_episode_complete(
                            episode + 1, 0.0, episode_steps, total_steps, infos
                        )
                    except Exception as e2:
                        print(f" 异常情况增强记录器失败: {e2}")

                episode += 1
                continue

            # 记录增强分析
            if (episode + 1) % 5 == 0:  # 每5个episode记录一次增强分析
                monitor.log_enhanced_analysis(env)

            # W&B记录
            if use_wandb:
                try:

                    step_reward = np.mean(list(rewards.values()))

                    # 基础奖励记录
                    log_data = {
                        "train/episode": int(episode + 1),
                        "train/episode_steps": int(episode_steps),
                        "train/total_steps": int(total_steps),
                        "train/episode_time": float(episode_time),
                        "global_step": int(total_steps)
                    }

                    # 记录不同尺度的奖励
                    # 1. 步奖励 (单步奖励)
                    log_data["rewards/step_reward"] = float(step_reward)

                    # 2. 累积奖励 (Episode总奖励)
                    log_data["rewards/episode_reward_total"] = float(episode_reward)

                    # 3. 标准化奖励
                    normalized_episode_reward = episode_reward / episode_steps if episode_steps > 0 else 0
                    log_data["rewards/episode_reward_normalized"] = float(normalized_episode_reward)

                    # 4. 日均奖励
                    daily_reward = episode_reward / (episode_steps / 24.0) if episode_steps > 0 else 0
                    log_data["rewards/daily_reward"] = float(daily_reward)

                    # 5. 小时奖励
                    hourly_reward = episode_reward / episode_steps if episode_steps > 0 else 0
                    log_data["rewards/hourly_reward"] = float(hourly_reward)


                    if 'infos' in locals() and infos:

                        first_agent_info = None
                        for agent_id, info in infos.items():
                            if isinstance(info, dict):
                                first_agent_info = info
                                break

                        if first_agent_info:
                            # 性能指标
                            if 'supply_satisfaction_rate' in first_agent_info:
                                log_data["metrics/supply_satisfaction"] = float(
                                    first_agent_info['supply_satisfaction_rate'])

                            if 'avg_reservoir_level' in first_agent_info:
                                log_data["metrics/reservoir_safety"] = float(
                                    first_agent_info['avg_reservoir_level'])

                            # 奖励组件分解
                            if 'reward_components' in first_agent_info:
                                reward_components = first_agent_info['reward_components']
                                if isinstance(reward_components, dict):
                                    # 原始奖励组件
                                    for component, value in reward_components.items():
                                        if isinstance(value, (int, float)) and not (isinstance(value, float) and (
                                                math.isnan(value) or math.isinf(value))):
                                            log_data[f"rewards/components/{component}"] = float(value)

                                    total_supply_reward = reward_components.get('supply', 0) * episode_steps
                                    total_safety_reward = reward_components.get('safety', 0) * episode_steps
                                    total_ecological_reward = reward_components.get('ecological', 0) * episode_steps

                                    log_data["rewards/cumulative/supply"] = float(total_supply_reward)
                                    log_data["rewards/cumulative/safety"] = float(total_safety_reward)
                                    log_data["rewards/cumulative/ecological"] = float(total_ecological_reward)


                                    total_component_reward = sum(abs(v) for v in reward_components.values())
                                    if total_component_reward > 0:
                                        for component, value in reward_components.items():
                                            weight = abs(value) / total_component_reward
                                            log_data[f"rewards/weights/{component}"] = float(weight)

                            # 智能体级别的奖励分解
                            agent_rewards = {}
                            for agent_id, info in infos.items():
                                if isinstance(info, dict) and 'total_reward' in info:
                                    agent_rewards[agent_id] = info['total_reward']

                            if agent_rewards:
                                # 智能体奖励统计
                                agent_reward_values = list(agent_rewards.values())
                                log_data["rewards/agents/mean"] = float(np.mean(agent_reward_values))
                                log_data["rewards/agents/std"] = float(np.std(agent_reward_values))
                                log_data["rewards/agents/max"] = float(np.max(agent_reward_values))
                                log_data["rewards/agents/min"] = float(np.min(agent_reward_values))

                                # 个体智能体奖励
                                for agent_id, reward in agent_rewards.items():
                                    log_data[f"rewards/individual/{agent_id}"] = float(reward)

                            # 训练阶段
                            if 'phase' in first_agent_info:
                                log_data["training/phase"] = str(first_agent_info['phase'])

                            # 探索状态
                            if 'exploration_active' in first_agent_info:
                                log_data["training/exploration_active"] = int(
                                    bool(first_agent_info['exploration_active']))

                            if 'diversity_score' in first_agent_info:
                                diversity_val = first_agent_info['diversity_score']
                                if isinstance(diversity_val, (int, float)) and not (
                                        isinstance(diversity_val, float) and (
                                        math.isnan(diversity_val) or math.isinf(diversity_val))):
                                    log_data["training/diversity_score"] = float(diversity_val)

                        # 奖励趋势分析
                        if hasattr(monitor, 'episode_rewards') and len(monitor.episode_rewards) > 10:
                            recent_rewards = list(monitor.episode_rewards)[-10:]
                            log_data["rewards/trends/mean_10"] = float(np.mean(recent_rewards))
                            log_data["rewards/trends/std_10"] = float(np.std(recent_rewards))

                            if len(recent_rewards) >= 5:
                                first_half = np.mean(recent_rewards[:5])
                                second_half = np.mean(recent_rewards[5:])
                                trend = (second_half - first_half) / (abs(first_half) + 1e-8)
                                log_data["rewards/trends/direction"] = float(trend)

                        # 记录并验证
                        wandb.log(log_data)
                        print(f" wandb 记录成功: Episode {episode + 1}, 数据点: {len(log_data)}")

                        # 每10个episode记录聚合数据
                        if (episode + 1) % 10 == 0:
                            aggregated_data = {
                                "milestones/episodes_10": episode + 1,
                                "milestones/total_steps_10": total_steps,
                                "milestones/timestamp": time.time()
                            }

                            # 10个episode的聚合奖励统计
                            if hasattr(monitor, 'episode_rewards') and len(monitor.episode_rewards) >= 10:
                                last_10_rewards = list(monitor.episode_rewards)[-10:]
                                aggregated_data.update({
                                    "milestones/reward_mean_10": float(np.mean(last_10_rewards)),
                                    "milestones/reward_std_10": float(np.std(last_10_rewards)),
                                    "milestones/reward_max_10": float(np.max(last_10_rewards)),
                                    "milestones/reward_min_10": float(np.min(last_10_rewards))
                                })

                            wandb.log(aggregated_data)

                except Exception as e:
                    print(f" W&B记录详细错误: {e}")
                    import traceback
                    print(f"完整错误堆栈: {traceback.format_exc()}")

                    # 环境指标
                    if infos:
                        first_agent_info = list(infos.values())[0]
                        if isinstance(first_agent_info, dict):
                            if 'supply_satisfaction_rate' in first_agent_info:
                                log_data["metrics/supply_satisfaction"] = float(
                                    first_agent_info['supply_satisfaction_rate'])
                            if 'avg_reservoir_level' in first_agent_info:
                                log_data["metrics/reservoir_safety"] = float(
                                    first_agent_info['avg_reservoir_level'])
                            if 'reward_components' in first_agent_info:
                                reward_components = first_agent_info['reward_components']
                                for component, value in reward_components.items():
                                    if isinstance(value, (int, float)):
                                        log_data[f"rewards/{component}"] = float(value)

                    wandb.log(log_data)
                    print(f" 强制 wandb 记录: Episode {episode + 1}, 数据点: {len(log_data)}")

                except Exception as e:
                    print(f" W&B记录详细错误: {e}")
                    import traceback
                    print(traceback.format_exc())

            # 内存清理
            if (episode + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 新增：定期保存历史数据（每100个Episode）
            if enhanced_logger and (episode + 1) % 100 == 0:
                try:
                    print(f"Episode {episode + 1} 触发定期保存...")
                    print(f"当前历史数据: Episodes={len(enhanced_logger.episode_history)}, Steps={len(enhanced_logger.step_history)}")
                    enhanced_logger.save_history_files()
                    print(f"定期保存历史数据: Episode {episode + 1}")

                    # 创建检查点兼容文件
                    checkpoint_file = enhanced_logger.create_wandb_compatible_history()
                    if checkpoint_file:
                        print(f"检查点文件已更新: {checkpoint_file}")
                    else:
                        print(f" 检查点文件创建失败")

                    #  保存模型检查点
                    current_avg_reward = np.mean(list(monitor.episode_rewards)[-10:]) if len(monitor.episode_rewards) >= 10 else 0.0
                    checkpoint_path = Path(args.run_dir) / f"checkpoint_episode_{episode + 1}.pt"
                    torch.save({
                        'actor_state_dict': policy.actor.state_dict(),
                        'critic_state_dict': policy.critic.state_dict(),
                        'episode': episode + 1,
                        'total_steps': total_steps,
                        'avg_reward': current_avg_reward,
                        'timestamp': time.time()
                    }, str(checkpoint_path))
                    print(f" 模型检查点已保存: {checkpoint_path.name}")

                except Exception as e:
                    print(f" 定期保存失败: {e}")
            elif enhanced_logger and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1} 检查点：历史长度 {len(enhanced_logger.episode_history)}, 距离保存还有 {100 - ((episode + 1) % 100)} episodes")
            elif not enhanced_logger and (episode + 1) % 100 == 0:
                print(f"️ Episode {episode + 1} 应该定期保存，but enhanced_logger is None")

            # 评估
            if (episode + 1) % args.eval_interval == 0:
                print(f"\n 第 {episode + 1} 次评估")
                try:
                    eval_reward = evaluate_optimized(policy, eval_env, args.eval_episodes, args.episode_length,
                                                     args)
                    monitor.log_eval(eval_reward)

                    if use_wandb:
                        try:
                            wandb.log({
                                "eval/reward": eval_reward,
                                "global_step": total_steps
                            })
                        except Exception as e:
                            print(f" W&B评估记录失败: {e}")

                        # 早停检查
                        if eval_reward > best_eval_reward:
                            best_eval_reward = eval_reward
                            no_improvement_count = 0

                            # 保存最佳模型
                            try:
                                save_path = Path(args.run_dir) / "best_model.pt"
                                torch.save({
                                    'actor_state_dict': policy.actor.state_dict(),
                                    'critic_state_dict': policy.critic.state_dict(),
                                    'episode': episode,
                                    'best_eval_reward': best_eval_reward,
                                    'total_steps': total_steps
                                }, str(save_path))
                                print(f" 保存最佳模型，奖励: {best_eval_reward:.3f}")
                            except Exception as e:
                                print(f"️ 保存模型失败: {e}")
                        else:
                            no_improvement_count += 1
                            print(f" 无改善: {no_improvement_count}/{max_no_improvement}")

                            if no_improvement_count >= max_no_improvement:
                                print(f" 早停触发")
                                training_completed = True
                                break

                except Exception as e:
                    print(f"️ 评估过程出错: {e}")

            episode += 1

    except KeyboardInterrupt:
        print("\n️ 训练被用户中断")
        training_completed = True

    finally:
        # 训练结束清理
        final_stats = monitor.get_stats()
        print(f"\n 训练结束")
        print(f" 最终统计: {final_stats}")

        #  保存增强记录器的完整历史数据
        if enhanced_logger:
            try:
                # 保存历史文件并清理临时文件
                enhanced_logger.save_history_files(clean_temp_files=True)

                # 创建wandb兼容的历史文件
                wandb_history_file = enhanced_logger.create_wandb_compatible_history()
                if wandb_history_file:
                    print(f" WandB兼容历史文件已创建: {wandb_history_file}")

                # 获取训练摘要
                summary = enhanced_logger.get_training_summary()
                print(f" 训练摘要:")
                print(f"   总Episodes: {summary.get('total_episodes', 0):,}")
                print(f"   总步数: {summary.get('total_steps', 0):,}")
                print(f"   最终奖励: {summary.get('final_reward', 0):.3f}")
                print(f"   最大奖励: {summary.get('max_reward', 0):.3f}")
                print(f"   平均奖励: {summary.get('avg_reward', 0):.3f}")
                print(f"   训练时长: {summary.get('training_duration', 0)/3600:.1f} 小时")

                # 摘要记录到wandb
                if use_wandb:
                    try:
                        wandb.log({
                            "final/total_episodes": summary.get('total_episodes', 0),
                            "final/total_steps": summary.get('total_steps', 0),
                            "final/final_reward": summary.get('final_reward', 0),
                            "final/max_reward": summary.get('max_reward', 0),
                            "final/avg_reward": summary.get('avg_reward', 0),
                            "final/training_duration_hours": summary.get('training_duration', 0)/3600
                        })
                        print(" 训练摘要已记录到wandb")
                    except Exception as e:
                        print(f" 训练摘要wandb记录失败: {e}")

            except Exception as e:
                print(f" 增强记录器数据保存失败: {e}")

        # 清理资源
        monitor.cleanup()

        if use_wandb:
            try:
                wandb.log({"training_completed": True, "final_stats": final_stats})
            except Exception as e:
                print(f"️ 最终W&B记录失败: {e}")


def evaluate_optimized(policy, env, num_episodes, episode_length, args):
    """评估函数"""
    eval_rewards = []

    for eval_ep in range(num_episodes):
        try:
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0

            rnn_states_actor = np.zeros((args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
            masks = np.ones((args.num_agents, 1), dtype=np.float32)

            while episode_steps < episode_length:
                try:
                    with torch.no_grad():
                        obs_array = np.array([obs[agent] for agent in env.agents], dtype=np.float32)
                        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(args.device)
                        rnn_states_actor_tensor = torch.from_numpy(rnn_states_actor).unsqueeze(0).to(args.device)
                        masks_tensor = torch.from_numpy(masks).unsqueeze(0).to(args.device)

                        action, rnn_states_actor_new = policy.act(
                            obs_tensor,
                            rnn_states_actor_tensor,
                            masks_tensor,
                            deterministic=True
                        )

                        rnn_states_actor = safe_tensor_to_numpy(rnn_states_actor_new.squeeze(0))
                        actions_np = safe_tensor_to_numpy(action.squeeze(0))
                        actions = convert_actions_from_mappo(env, actions_np)

                    next_obs, rewards, dones, truncations, infos = env.step(actions)

                    episode_reward += np.mean(list(rewards.values()))
                    episode_steps += 1

                    # 更新masks
                    masks = np.ones((args.num_agents, 1), dtype=np.float32)
                    for i, agent_id in enumerate(env.agents):
                        if dones[agent_id] or any(t for t in truncations.values() if t):
                            masks[i] = 0.0

                    if all(dones.values()) or any(truncations.values()):
                        break

                    obs = next_obs

                except Exception as e:
                    print(f"️ 评估步骤出错: {e}")
                    break

            eval_rewards.append(episode_reward)

        except Exception as e:
            print(f"️ 评估episode {eval_ep + 1} 出错: {e}")
            continue

    return np.mean(eval_rewards) if eval_rewards else 0.0


def get_default_args():
    """获取默认参数配置"""
    parser = get_config()

    return parser


def main():
    """兼容wandb agent和直接运行"""
    print(" 开始水资源管理训练")
    print("=" * 60)

    try:
        # 获取参数配置
        parser = get_default_args()

        # 添加优化功能相关参数
        if not any(arg for arg in parser._option_string_actions if arg == '--enable_optimizations'):
            parser.add_argument('--enable_optimizations', action='store_true', default=False,
                               help='启用优化功能')

        # 添加wandb项目参数
        if not any(arg for arg in parser._option_string_actions if arg == '--wandb_project'):
            parser.add_argument('--wandb_project', type=str, default='uncategorized',
                               help='WandB项目名称（默认: uncategorized）')

        # 默认启用WandB和数据记录
        if not any(arg for arg in parser._option_string_actions if arg == '--use_wandb'):
            parser.add_argument('--use_wandb', action='store_true', default=True,
                               help='启用WandB记录（现在默认启用）')

        # 解析参数
        args = parser.parse_args()

        # 添加调试信息
        print(f"WandB配置检查:")
        print(f"   - use_wandb: {args.use_wandb}")
        print(f"   - ENHANCED_LOGGER_AVAILABLE: {ENHANCED_LOGGER_AVAILABLE}")
        print(f"   - wandb_project: {getattr(args, 'wandb_project', 'uncategorized')}")

        # 设置设备
        if hasattr(args, 'cuda') and args.cuda and torch.cuda.is_available():
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")

        # 设置智能体数量（基于环境参数）
        args.num_agents = args.num_reservoirs + args.num_plants

        # 设置运行目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_name = "water_optimized" if args.enable_optimizations else "water_standard"
        args.run_dir = f"./results/{project_name}_{timestamp}"
        os.makedirs(args.run_dir, exist_ok=True)

        # 应用优化超参数（如果启用了优化）
        if args.enable_optimizations:
            if CONFIG_AVAILABLE:
                optimized_params = get_optimized_hyperparameters()

                # 更新关键参数
                args.hidden_size = 128
                args.entropy_coef = 0.02
                args.lr = optimized_params.get('lr', 4e-5)
                args.critic_lr = optimized_params.get('critic_lr', 1.5e-4)
                args.clip_param = optimized_params.get('clip_param', 0.05)
                args.ppo_epoch = optimized_params.get('ppo_epoch', 4)
                args.max_grad_norm = optimized_params.get('max_grad_norm', 0.5)

                print(f" 应用优化配置:")
                print(f"   - 隐藏层大小: {args.hidden_size}")
                print(f"   - 学习率: {args.lr}")
                print(f"   - 熵系数: {args.entropy_coef}")
            else:
                print("️ 配置文件不可用，使用默认优化参数")

        # 显示WandB配置
        if args.use_wandb:
            print(f" WandB配置:")
            print(f"   - 项目名: {args.wandb_project}")

        # 打印配置摘要
        if CONFIG_AVAILABLE:
            print_config_summary(args)
        else:
            print(f"   - 智能体: {args.num_reservoirs}水库 + {args.num_plants}水厂")
            print(f"   - Episode长度: {args.episode_length}")
            print(f"   - 总步数: {args.num_env_steps}")
            print(f"   - 学习率: {args.lr}")
            print(f"   - 设备: {args.device}")

        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # 开始训练
        print("\n 开始训练...")
        train(args)

        print("\n 训练完成!")

    except KeyboardInterrupt:
        print("\n️ 训练被用户中断")
    except Exception as e:
        print(f"\n 训练出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        try:
            if 'args' in locals() and args.use_wandb:
                wandb.finish()
        except:
            pass

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n 训练结束")


if __name__ == "__main__":
    main()
else:
    # 兼容 wandb agent
    print(" 水资源管理训练模块已导入")
    print(" wandb agent 超参数调优")
    