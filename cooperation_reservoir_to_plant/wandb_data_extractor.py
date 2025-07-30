"""
WandB数据提取器
用于记录和保存训练历史数据
"""

import os
import json
import time
from pathlib import Path
from collections import deque
import numpy as np

class EnhancedWandBLogger:
    """WandB记录器 - 提供完整的历史数据保存功能"""

    def __init__(self, log_dir="./wandb_data", backup_interval=100, max_backups=5):
        """初始化增强WandB记录器

        参数:
            log_dir: 日志保存目录
            backup_interval: 增量备份间隔(episodes)
            max_backups: 保留的最大备份数量
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 配置
        self.backup_interval = backup_interval
        self.max_backups = max_backups

        # 历史数据存储
        self.episode_history = []
        self.step_history = []
        self.wandb_history = []

        # 训练统计
        self.start_time = time.time()
        self.total_episodes = 0
        self.total_steps = 0

        print(f"WandB记录器初始化: {self.log_dir}")
        print(f"   - 备份间隔: {backup_interval} episodes")
        print(f"   - 最大备份数: {max_backups}")
    
    def log_episode_complete(self, episode, episode_reward, episode_steps, total_steps, infos):
        """记录完整的episode数据"""
        try:
            timestamp = time.time()
            
            # 提取关键信息
            first_agent_info = list(infos.values())[0] if infos else {}
            
            # Episode数据
            episode_data = {
                "episode": episode,
                "episode_reward": float(episode_reward),
                "episode_steps": episode_steps,
                "total_steps": total_steps,
                "timestamp": timestamp,
                "step_per_episode": float(episode_steps),
                "_step": total_steps
            }
            
            # 添加性能指标
            if isinstance(first_agent_info, dict):
                if 'supply_satisfaction_rate' in first_agent_info:
                    episode_data['supply_satisfaction'] = float(first_agent_info['supply_satisfaction_rate'])
                if 'avg_reservoir_level' in first_agent_info:
                    episode_data['reservoir_safety'] = float(first_agent_info['avg_reservoir_level'])
                
                # 奖励组件
                if 'reward_components' in first_agent_info:
                    reward_components = first_agent_info['reward_components']
                    if isinstance(reward_components, dict):
                        for component, value in reward_components.items():
                            if isinstance(value, (int, float)) and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                                episode_data[f'reward_{component}'] = float(value)
                
                # 训练状态
                if 'phase' in first_agent_info:
                    episode_data['phase'] = str(first_agent_info['phase'])
                if 'exploration_active' in first_agent_info:
                    episode_data['exploration_active'] = bool(first_agent_info['exploration_active'])
            
            # 保存到历史记录
            self.episode_history.append(episode_data)
            self.wandb_history.append(episode_data)
            
            self.total_episodes = episode
            self.total_steps = total_steps
            
            # 每100个episode自动保存一次
            if episode % 100 == 0 and episode > 0:
                self._save_incremental_data()
                
        except Exception as e:
            print(f" Episode记录失败: {e}")
    
    def log_step_data(self, step, step_reward, action_diversity=None):
        """记录step级别数据"""
        try:
            step_data = {
                "step": step,
                "step_reward": float(step_reward),
                "timestamp": time.time(),
                "_step": step
            }
            
            if action_diversity is not None:
                step_data["action_diversity"] = float(action_diversity)
            
            self.step_history.append(step_data)
            
        except Exception as e:
            print(f" Step记录失败: {e}")

    def save_history_files(self, clean_temp_files=False):
        """保存历史文件，可选是否清理临时文件"""
        try:
            # 保存episode历史
            episode_file = self.log_dir / "episode_history.json"
            with open(episode_file, 'w', encoding='utf-8') as f:
                json.dump(self.episode_history, f, indent=2, ensure_ascii=False)

            # 保存step历史
            step_file = self.log_dir / "step_history.json"
            with open(step_file, 'w', encoding='utf-8') as f:
                json.dump(self.step_history, f, indent=2, ensure_ascii=False)

            # 保存wandb格式历史
            wandb_file = self.log_dir / "wandb-history.jsonl"
            with open(wandb_file, 'w', encoding='utf-8') as f:
                for entry in self.wandb_history:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f" 历史文件已保存到: {self.log_dir}")
            print(f"   - Episodes: {len(self.episode_history)}")
            print(f"   - Steps: {len(self.step_history)}")

            # 清理冗余临时文件
            if clean_temp_files:
                self._clean_temp_files()

            return True

        except Exception as e:
            print(f"保存历史文件失败: {e}")
            return False
    
    def create_wandb_compatible_history(self):
        """创建WandB兼容的历史文件"""
        try:
            wandb_file = self.log_dir / "wandb-history.jsonl"
            with open(wandb_file, 'w', encoding='utf-8') as f:
                for entry in self.wandb_history:
                    # 添加WandB格式的时间戳
                    entry_copy = entry.copy()
                    entry_copy['_timestamp'] = entry_copy.get('timestamp', time.time())
                    f.write(json.dumps(entry_copy, ensure_ascii=False) + '\n')
            
            print(f" WandB兼容文件已创建: {wandb_file}")
            return str(wandb_file)
            
        except Exception as e:
            print(f" 创建WandB兼容文件失败: {e}")
            return None
    
    def get_training_summary(self):
        """获取训练摘要"""
        if not self.episode_history:
            return {
                'total_episodes': 0,
                'total_steps': 0,
                'training_duration': 0,
                'final_reward': 0,
                'max_reward': 0,
                'avg_reward': 0
            }
        
        episode_rewards = [ep.get('episode_reward', 0) for ep in self.episode_history]
        training_duration = time.time() - self.start_time
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'training_duration': training_duration,
            'final_reward': episode_rewards[-1] if episode_rewards else 0,
            'max_reward': max(episode_rewards) if episode_rewards else 0,
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0
        }

    def _save_incremental_data(self):
        """保存数据 - 滚动备份"""
        try:
            # 保存最近的数据到临时文件
            recent_episodes = self.episode_history[-100:] if len(self.episode_history) >= 100 else self.episode_history

            # 使用固定文件名而不是时间戳，减少文件数量
            backup_id = self.total_episodes // 100 % 5  # 只保留5个滚动备份
            temp_file = self.log_dir / f"recent_episodes_backup_{backup_id}.json"

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(recent_episodes, f, indent=2, ensure_ascii=False)

            print(f"增量备份已保存: {temp_file.name} (Episodes {self.total_episodes - 100}-{self.total_episodes})")

        except Exception as e:
            print(f" 增量保存失败: {e}")

    def _clean_temp_files(self):
        """清理旧的临时文件，保留滚动备份"""
        try:
            # 查找使用旧命名格式的临时文件
            old_pattern_files = list(self.log_dir.glob("recent_episodes_[0-9]*.json"))
            if old_pattern_files:
                for old_file in old_pattern_files:
                    old_file.unlink()
                print(f" 已清理 {len(old_pattern_files)} 个旧格式临时文件")
        except Exception as e:
            print(f" 清理临时文件失败: {e}")

WandBDataExtractor = EnhancedWandBLogger

if __name__ == "__main__":
    print("增强WandB数据提取器模块")
    print("用于记录和保存训练历史数据") 