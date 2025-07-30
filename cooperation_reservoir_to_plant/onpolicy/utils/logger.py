import os
import json
import time
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from datetime import datetime

class Logger:
    """训练日志记录器"""
    
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建指标记录字典
        self.metrics = {
            'episode_rewards': [],
            'eval_rewards': [],
            'water_satisfaction': [],
            'reservoir_safety': [],
            'flood_risk': [],
            'drought_risk': [],
            'system_efficiency': [],
            'subgoals': [],
            'losses': {
                'policy_loss': [],
                'value_loss': [],
                'entropy_loss': []
            },
            'learning_rates': []
        }
        
        # 初始化wandb（如果启用）
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "water-management-mappo"),
                config=config,
                name=f"water_mappo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                entity=config.get("wandb_entity")
            )
        
        # 记录配置
        self._save_config()
    
    def _save_config(self):
        """保存配置文件"""
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def log_metrics(self, metrics_dict, step):
        """记录训练指标"""
        # 更新本地指标记录
        for key, value in metrics_dict.items():
            if key in self.metrics:
                if isinstance(value, (list, np.ndarray)):
                    self.metrics[key].extend(value)
                else:
                    self.metrics[key].append(value)
            elif key in self.metrics['losses']:
                self.metrics['losses'][key].append(value)
        
        # 记录到wandb（如果启用）
        if self.use_wandb:
            wandb.log(metrics_dict, step=step)
        
        # 定期保存指标到文件
        if step % self.config.get("log_interval", 10) == 0:
            self._save_metrics()
    
    def log_eval(self, eval_metrics, step):
        """记录评估指标"""
        # 更新评估指标
        for key, value in eval_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # 记录到wandb（如果启用）
        if self.use_wandb:
            wandb.log({f"eval_{k}": v for k, v in eval_metrics.items()}, step=step)
    
    def log_model(self, model, optimizer, step, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': self.metrics
        }
        
        # 保存最新模型
        latest_path = self.log_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = self.log_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def _save_metrics(self):
        """保存指标到文件"""
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def plot_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 10))
        
        # 绘制奖励曲线
        plt.subplot(2, 3, 1)
        plt.plot(self.metrics['episode_rewards'], label='Training Reward')
        plt.plot(self.metrics['eval_rewards'], label='Evaluation Reward')
        plt.title('Training and Evaluation Rewards')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        
        # 绘制供水满足率和水库安全运行率
        plt.subplot(2, 3, 2)
        plt.plot(self.metrics['water_satisfaction'], label='Water Satisfaction')
        plt.plot(self.metrics['reservoir_safety'], label='Reservoir Safety')
        plt.title('Water Supply Metrics')
        plt.xlabel('Step')
        plt.ylabel('Rate')
        plt.legend()
        
        # 绘制风险指标
        plt.subplot(2, 3, 3)
        plt.plot(self.metrics['flood_risk'], label='Flood Risk')
        plt.plot(self.metrics['drought_risk'], label='Drought Risk')
        plt.title('Risk Indicators')
        plt.xlabel('Step')
        plt.ylabel('Risk Level')
        plt.legend()
        
        # 绘制系统效率
        plt.subplot(2, 3, 4)
        plt.plot(self.metrics['system_efficiency'], label='System Efficiency')
        plt.title('System Efficiency')
        plt.xlabel('Step')
        plt.ylabel('Efficiency')
        plt.legend()
        
        # 绘制损失曲线
        plt.subplot(2, 3, 5)
        for loss_name, loss_values in self.metrics['losses'].items():
            plt.plot(loss_values, label=loss_name)
        plt.title('Training Losses')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制子目标
        plt.subplot(2, 3, 6)
        subgoals = self.metrics['subgoals']
        plt.plot([sg['target_level'] for sg in subgoals], label='Target Level')
        plt.plot([sg['target_release'] for sg in subgoals], label='Target Release')
        plt.plot([sg['target_safety'] for sg in subgoals], label='Target Safety')
        plt.title('Subgoals')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / "training_curves.png")
        plt.close()
    
    def close(self):
        """关闭日志记录器"""
        # 保存最终指标
        self._save_metrics()
        
        # 绘制最终训练曲线
        self.plot_curves()
        
        # 关闭wandb（如果启用）
        if self.use_wandb:
            wandb.finish() 