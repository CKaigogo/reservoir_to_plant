# reservoir_to_plant
# 水资源合作管理环境 (WaterManagementEnv) 使用说明

## 运行环境与依赖库

```bash
# 推荐使用Conda创建虚拟环境
conda create -n water_env python=3.10
conda activate water_env

# 安装依赖
pip install torch numpy matplotlib wandb gym seaborn scipy
```

**注意：如需使用CUDA加速，请安装对应的PyTorch GPU版本**

## 环境概述

水资源管理环境是一个多智能体强化学习环境，模拟水库-水厂系统的水资源调度与管理问题。该环境支持多水库、多水厂的复杂水系统建模，可用于训练智能体优化水资源分配、平衡供需关系、保障生态安全等极端情况。

## 预训练模型

本环境采用深度强化学习方法MAPPO作为神经网络策略模型架构，预训练模型位于`wandb/run_final/files`目录下。

## 快速开始

### 基本训练流程

```python
from onpolicy.envs.water_env.WaterManagementEnv import WaterManagementEnv
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

# 创建环境
env = WaterManagementEnv(
    num_reservoirs=15,
    num_plants=6,
    use_fixed_connections=True,
    continuous_management=False,
    max_episode_steps=96,
    enable_optimizations=True
)

# 环境重置
obs = env.reset()

# 环境交互
for step in range(100):
    # 示例动作，实际使用时应从策略网络获取
    actions = {
        'reservoir_0': {
            'total_release_ratio': [0.3],
            'allocation_weights': [0.7, 0.3],
            'emergency_release': 0
        },
        'plant_0': {
            'demand_adjustment': [1.0],
            'priority_level': 1,
            'storage_strategy': [0.5]
        },
        # ... 其他智能体的动作
    }
    
    # 环境步进
    next_obs, rewards, dones, truncations, infos = env.step(actions)
    
    # 更新观测
    obs = next_obs
    
    # 检查是否结束
    if all(dones.values()):
        break
```

### 使用训练脚本

```bash
# 标准训练
python train_water.py --num_reservoirs 15 --num_plants 6 --episode_length 96

```

## 环境参数

### 构造函数参数

`WaterManagementEnv(num_reservoirs=15, num_plants=6, max_episode_steps=96, continuous_management=False, progressive_training_manager=None, use_fixed_connections=True, enable_optimizations=True)`

- `num_reservoirs`：水库数量，默认为15
- `num_plants`：水厂数量，默认为6
- `max_episode_steps`：每个episode的最大步数，默认为96（相当于4天，每一步为一小时）
- `progressive_training_manager`：渐进式训练管理器，默认为None
- `use_fixed_connections`：是否使用固定的水库-水厂连接关系，默认为True


##=====================================================================================================================


- `fixed_connections` = np.array([
- [True, False, False, False, False, False],   # reservoir_0 -> plant_0
- [False, False, True, False, False, False],   # reservoir_1 -> plant_2
- [False, False, True, False, False, False],   # reservoir_2 -> plant_2
- [False, True, False, False, False, False],   # reservoir_3 -> plant_1
- [False, True, False, False, False, False],   # reservoir_4 -> plant_1
- [False, True, False, False, False, False],   # reservoir_5 -> plant_1
- [False, False, False, True, False, False],
- [False, False, False, False, True, False],
- [True, False, False, False, False, False],   # reservoir_8 -> plant_0
- [False, False, False, False, False, True],
- [True, False, False, False, False, False],   # reservoir_10 -> plant_0
- [True, False, False, False, False, False],   # reservoir_11 -> plant_0
- [True, False, False, False, False, False],   # reservoir_12 -> plant_0
- [True, False, False, False, False, False],   # reservoir_13 -> plant_0
- [False, True, False, False, False, False],   # reservoir_14 -> plant_1
- ])
##=====================================================================================================================
- `enable_optimizations`：是否启用训练优化，默认为True

## 智能体与观测空间

环境中的智能体分为两类：

1. **水库智能体**：负责控制水库的放水量和分配策略
2. **水厂智能体**：负责调整需水量和存储策略

### 观测空间

每个智能体的观测空间包含以下信息：

#### 水库智能体观测

- 水库自身状态（当前水位、最大容量、入流量等）
- 连接的水厂信息（需水量、优先级等）
- 全局状态（时间信息、天气条件等）
- 历史数据（过去几个时间步的状态）
- 相邻水库状态（如果启用）

#### 水厂智能体观测

- 水厂自身状态（当前存水量、需水量、存储容量等）
- 连接的水库信息（当前水位、供水能力等）
- 全局状态（时间信息、天气条件等）
- 历史数据（过去几个时间步的状态）
- 竞争水厂状态（当前情况下无竞争水厂）

## 动作空间

### 水库智能体动作

```python
{
    'total_release_ratio': [0.0-1.0],  # 总放水比例
    'allocation_weights': [w1, w2, ...],  # 分配给各水厂的权重
    'emergency_release': 0 or 1  # 是否进行应急放水
}
```

### 水厂智能体动作

```python
{
    'demand_adjustment': [0.5-1.5],  # 需水量调整系数
    'priority_level': 0, 1, or 2     # 优先级别
    'storage_strategy': [0.0-1.0]    # 存储策略
}
```

## 奖励系统

环境使用多组件奖励系统，包括：

- **供水奖励**：基于满足水厂需求的程度
- **安全奖励**：基于水库水位维持在安全范围内
- **生态奖励**：基于满足生态流量要求的程度
- **稳定性奖励**：基于动作的稳定性和平滑度

## 优化功能（默认不启用）

通过设置`enable_optimizations=True`可以启用以下优化：


## 训练监控

环境支持与WandB集成，可以记录和可视化训练过程中的各种指标：

- 奖励分解（供水、安全、生态等组件）
- 水库水位状态
- 水厂需求满足率
- 策略网络参数统计
- reward迭代变化情况


## 示例代码

### 训练示例

详见`train_water.py`文件，该文件提供了完整的训练流程，包括：

- 环境创建与配置
- 策略网络初始化
- 训练循环
- WandB集成
- 模型保存与评估

### 超参数调优

详见`start_hyperparameter_tuning.py`文件，该文件提供了使用WandB进行超参数调优的示例。


## API参考

详细的API参考请查看源代码文档：
- `WaterManagementEnv`：主环境类
- `ActionProcessor`：动作处理器
- `StructuredObservationManager`：观测管理器
- `StabilizedMultiAgentRewardSystem`：奖励设置
- `PotentialBasedRewardShaper`：基于势函数的奖励塑形器

### 核心环境类接口

#### WaterManagementEnv

```python
WaterManagementEnv(
    num_reservoirs=15,           # 水库数量
    num_plants=6,                # 水厂数量
    max_episode_steps=96,        # 最大步数（默认为4天，每小时一步）
    progressive_training_manager=None, # 渐进式训练管理器
    use_fixed_connections=True,  # 是否使用固定连接关系
    enable_optimizations=True    # 是否启用训练优化
)
```

##### 主要方法

| 方法名 | 参数 | 返回值 | 说明 |
|-------|------|-------|------|
| `reset(seed=None, options=None)` | `seed`: 随机种子| `observations`: 初始观测字典 | 重置环境并返回初始观测 |
| `step(actions)` | `actions`: 动作字典 | `observations`: 新观测<br>`rewards`: 奖励字典<br>`dones`: 结束标志字典<br>`truncations`: 截断标志字典<br>`infos`: 信息字典 | 执行一步环境交互 |
| `seed(seed=None)` | `seed`: 随机种子 | `seeds`: 使用的种子列表 | 设置随机种子 |
| `render(mode="human")` | `mode`: 渲染模式 | 无 | 渲染当前环境状态 

##### 状态和信息获取

| 方法名 | 参数 | 返回值 | 说明 |
|-------|------|-------|------|
| `get_reward_system_status()` | 无 | 奖励系统状态字典 | 获取当前奖励系统状态 |
| `print_reward_analysis(detailed=False)` | `detailed`: 是否显示详细信息 | 无 | 打印奖励系统分析 |
| `get_optimization_status()` | 无 | 优化状态字典 | 获取当前环境的优化状态信息 |

##### 属性

| 属性名 | 类型 | 说明 |
|-------|------|------|
| `observation_spaces` | Dict | 各智能体的观测空间 |
| `action_spaces` | Dict | 各智能体的动作空间 |
| `state` | ndarray | 环境的全局状态向量 |
| `reservoirs` | ndarray | 当前各水库水量 |
| `max_reservoir` | ndarray | 水库最大容量 |
| `plants_demand` | ndarray | 水厂当前需水量 |
| `plant_inventory` | ndarray | 水厂当前库存水量 |
| `connections` | ndarray | 水库-水厂连接矩阵 |

### 辅助类接口

#### ActionProcessor

处理智能体动作，提供动作验证和转换功能。

```python
ActionProcessor(num_reservoirs, num_plants, connections)
```

##### 主要方法

| 方法名 | 参数 | 返回值 | 说明 |
|-------|------|-------|------|
| `rehydrate_actions_fixed(raw_actions)` | `raw_actions`: 原始动作 | `hydrated_actions`: 标准格式动作 | 将各种格式的动作转换为标准格式 |
| `_standardize_dict_action(agent_id, action_dict)` | `agent_id`: 智能体ID<br>`action_dict`: 动作字典 | `standardized_action`: 标准化后的动作 | 标准化字典格式动作 |
| `_get_safe_default_action(agent_id)` | `agent_id`: 智能体ID | `default_action`: 默认动作 | 获取安全的默认动作 |

#### StructuredObservationManager

管理观测空间，为智能体提供结构化观测。

```python
StructuredObservationManager(num_reservoirs, num_plants, connections)
```

##### 主要方法

| 方法名 | 参数 | 返回值 | 说明 |
|-------|------|-------|------|
| `get_structured_observation(agent_id, env_state)` | `agent_id`: 智能体ID<br>`env_state`: 环境状态 | `observation`: 结构化观测 | 获取指定智能体的结构化观测 |
| `update_history(env_state)` | `env_state`: 环境状态 | 无 | 更新历史观测数据 |
| `reset_history()` | 无 | 无 | 重置历史观测数据 |

#### StabilizedMultiAgentRewardSystem

标准奖励系统，提供多组件奖励计算。

```python
StabilizedMultiAgentRewardSystem(
    n_reservoirs,     # 水库数量
    n_plants,         # 水厂数量
    max_episode_steps, # 最大步数
    distance_matrix=None, # 距离矩阵
    cost_factor=0.01   # 成本系数
)
```

##### 主要方法

| 方法名 | 参数 | 返回值 | 说明 |
|-------|------|-------|------|
| `calculate_rewards(state, actions, current_step, episode_progress, is_terminal, prev_state)` | `state`: 当前状态<br>`actions`: 动作<br>`current_step`: 当前步数<br>`episode_progress`: 进度<br>`is_terminal`: 是否终止<br>`prev_state`: 前一状态 | `rewards`: 奖励字典<br>`infos`: 信息字典 | 计算当前步骤奖励 |
| `_calculate_supply_rewards(state)` | `state`: 当前状态 | 供水奖励字典 | 计算供水满足度奖励 |
| `_calculate_safety_rewards(state, conservation_actions)` | `state`: 当前状态<br>`conservation_actions`: 保护动作 | 安全奖励字典 | 计算水库安全水位奖励 |
| `_calculate_ecological_rewards(state)` | `state`: 当前状态 | 生态奖励字典 | 计算生态流量奖励 |

### 数据结构

#### 环境状态结构

```python
env_state = {
    'reservoirs': ndarray,         # 水库当前水量
    'max_reservoir': ndarray,      # 水库最大容量
    'plants_demand': ndarray,      # 水厂需水量
    'plant_inventory': ndarray,    # 水厂库存水量
    'current_hour': int,           # 当前小时
    'season_progress': float,      # 季节进度 [0-1]
    'total_steps': int,            # 总步数
    'dead_capacity': ndarray,      # 死水位容量
    'hourly_demand': ndarray,      # 每小时需水量
    'actual_supply': ndarray,      # 实际供水量
    'max_plant': ndarray,          # 水厂最大需水量
    'plant_storage_capacity': ndarray, # 水厂存储容量
    'target_ecological_flows': ndarray, # 目标生态流量
    'ecological_releases': ndarray  # 实际生态流量
}
```

#### 返回信息结构

```python
info = {
    # 基础系统信息
    "state": ndarray,                      # 全局状态向量
    "supply_satisfaction_rate": float,     # 供水满足率
    "avg_reservoir_level": float,          # 平均水库水位
    "system_efficiency": float,            # 系统效率
    "current_hour": int,                   # 当前小时
    "current_day": int,                    # 当前天
    
    # 水文信息
    "inflows": list,                       # 入流量
    "water_released": list,                # 放水量
    "actual_supply": list,                 # 实际供水量
    "ecological_releases": list,           # 生态放水量
    "target_ecological_flows": list,       # 目标生态流量
    "forced_spills": list,                 # 强制泄洪量
    
    # 奖励组件
    "supply_reward": float,                # 供水奖励
    "safety_reward": float,                # 安全奖励
    "ecological_reward": float,            # 生态奖励
    "stability_reward": float,             # 稳定性奖励
    "cost_reward": float                   # 成本奖励
}


## 环境信息获取

### 获取环境状态

```python
# 获取水库状态
reservoirs = env.reservoirs  # 当前水库水量
max_reservoir = env.max_reservoir  # 水库最大容量
reservoir_levels = env.reservoirs / env.max_reservoir  # 水库水位比例

# 获取水厂状态
plants_demand = env.plants_demand  # 水厂需水量
plants_inventory = env.plants_inventory  # 水厂当前存水量
```

### 获取性能指标

```python
# 获取奖励系统状态
if hasattr(env, 'get_reward_system_status'):
    status = env.get_reward_system_status()
    print(f"供水满足率: {status.get('supply_satisfaction_rate', 0):.2f}")
    print(f"水库安全水位: {status.get('avg_reservoir_level', 0):.2f}")
    print(f"生态流量偏差: {status.get('ecological_flow_deviation', 0):.2f}")
```

## 例程

- `train_water.py`：提供了完整的训练流程示例
- `start_hyperparameter_tuning.py`：提供了超参数调优示例 