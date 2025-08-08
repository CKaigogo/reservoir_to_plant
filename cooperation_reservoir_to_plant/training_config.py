#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training configuration for the water resource management environment
"""

import argparse
import torch
import numpy as np


def get_config():
    """Training configuration for the environment"""

    parser = argparse.ArgumentParser(description='Water Resource Management Training Configuration')

    # === Basic Experiment Configuration ===
    parser.add_argument('--algorithm_name', type=str, default='mappo', help='Algorithm name')
    parser.add_argument('--experiment_name', type=str, default='water_management',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--cuda_deterministic', action='store_true', default=False,
                        help='CUDA deterministic computation')

    # === Environment Configuration ===
    parser.add_argument('--env_name', type=str, default='water', help='Environment name')
    parser.add_argument('--num_reservoirs', type=int, default=15, help='Number of reservoirs')
    parser.add_argument('--num_plants', type=int, default=6, help='Number of plants')
    parser.add_argument("--use_fixed_connections", type=bool, default=True)
    parser.add_argument('--continuous_management', action='store_true', default=True,
                        help='Enable continuous management mode')

    # === Episode Configuration ===
    parser.add_argument('--episode_length', type=int, default=96,
                        help='Episode length (hours) ')
    parser.add_argument('--max_episode_steps', type=int, default=96, help='Maximum steps')
    parser.add_argument('--use_proper_time_limits', action='store_true', default=True,
                        help='Use proper time limits')

    # === Training Scale Configuration ===
    parser.add_argument('--num_env_steps', type=int, default=int(1300000),
                        help='Total environment steps - increased to 3 million to adapt to complex environment')
    parser.add_argument('--n_rollout_threads', type=int, default=16,
                        help='Number of parallel environments - increased parallelism')
    parser.add_argument('--n_training_threads', type=int, default=1, help='Number of training threads')

    # === Learning Rate Configuration ===
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate - reduced to adapt to complex reward functions')
    parser.add_argument('--critic_lr', type=float, default=1.2e-4,
                        help='Critic learning rate - slightly higher than Actor')
    parser.add_argument('--use_linear_lr_decay', action='store_true', default=True,
                        help='Use linear learning rate decay')

    # === PPO Configuration ===
    parser.add_argument('--ppo_epoch', type=int, default=5,
                        help='PPO training epochs - increased to fully learn complex rewards')
    parser.add_argument('--num_mini_batch', type=int, default=4,
                        help='Number of mini-batches')
    parser.add_argument('--clip_param', type=float, default=0.04,
                        help='PPO clipping parameter')  # Clipping parameter, tightened to increase stability
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.02,
                        help='Entropy regularization coefficient - balance exploration and exploitation')
    parser.add_argument('--max_grad_norm', type=float, default=0.4,  # 10.0, stricter gradient clipping, reduced to 0.5
                        help='Gradient clipping')
    parser.add_argument('--eps', type=float, default=1e-5, help='Adam optimizer epsilon')
    parser.add_argument('--opti_eps', type=float, default=1e-5, help='Optimizer epsilon')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')

    # === GAE Configuration  ===
    parser.add_argument('--use_gae', action='store_true', default=True,
                        help='Use GAE')
    parser.add_argument('--gamma', type=float, default=0.995,
                        help='Discount factor - emphasize long-term returns')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda parameter')

    # === Network Architecture  ===
    parser.add_argument('--share_policy', action='store_true', default=True,
                        help='Agents share policy')
    parser.add_argument('--use_centralized_V', action='store_true', default=True,
                        help='Use centralized value function')
    parser.add_argument('--use_obs_instead_of_state', action='store_false', default=False,
                        help='Critic uses global state instead of observation')

    # === Network Structure Configuration ===
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden layer size - increased network capacity')
    parser.add_argument('--layer_N', type=int, default=3,
                        help='Number of network layers')
    parser.add_argument('--use_ReLU', action='store_true', default=True,
                        help='Use ReLU activation function')
    parser.add_argument('--use_orthogonal', action='store_true', default=True,
                        help='Use orthogonal initialization')
    parser.add_argument('--gain', type=float, default=0.01, help='Orthogonal initialization gain')

    # === Feature Processing Configuration ===
    parser.add_argument('--use_feature_normalization', action='store_true', default=True,
                        help='Use feature normalization - important: handle observations of different scales')
    parser.add_argument('--use_valuenorm', action='store_true', default=False,
                        help='Use value normalization')
    parser.add_argument('--use_popart', action='store_false', default=False,
                        help='Use PopArt normalization')

    # === Loss Function Configuration ===
    parser.add_argument('--use_huber_loss', action='store_true', default=True,
                        help='Use Huber loss')
    parser.add_argument('--use_value_active_masks', action='store_true', default=True,
                        help='Use value active masks')
    parser.add_argument('--use_policy_active_masks', action='store_true', default=True,
                        help='Use policy active masks')
    parser.add_argument('--huber_delta', type=float, default=5.0, help='Huber loss delta')

    # === Evaluation Configuration ===
    parser.add_argument('--use_eval', action='store_true', default=True,
                        help='Enable evaluation')
    parser.add_argument('--max_episodes', type=int, default=15000,
                        help='Maximum training episodes')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluation interval - evaluate every 25 episodes')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Number of episodes per evaluation')
    parser.add_argument('--eval_stochastic', action='store_false', default=False,
                        help='Use deterministic policy during evaluation')

    # === Saving and Logging Configuration ===
    parser.add_argument('--save_interval', type=int, default=25,
                        help='Model saving interval')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log printing interval')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Use WandB for logging')
    parser.add_argument('--user_name', type=str, default='water_management',
                        help='User name')

    # === Advanced Configuration ===
    parser.add_argument('--use_recurrent_policy', action='store_false', default=False,
                        help='Use recurrent policy')
    parser.add_argument('--recurrent_N', type=int, default=1, help='Number of recurrent layers')
    parser.add_argument('--data_chunk_length', type=int, default=10, help='Data chunk length')

    # === Buffer and Sampling Configuration e ===
    parser.add_argument('--buffer_size', type=int, default=2048,
                        help='Buffer size - 8 episodes (168*8=1344) for experience sampling')
    parser.add_argument('--rollout_length', type=int, default=96,
                        help='Rollout length - consistent with episode_length')
    parser.add_argument('--use_rollout_active_masks', action='store_true', default=True,
                        help='Use rollout active masks')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size - matched with episode length')

    # === Reward Processing and Normalization ===
    parser.add_argument('--use_reward_normalization', action='store_true', default=True,
                        help='Use reward normalization - important: handle multiple reward components')
    parser.add_argument('--use_advantage_normalization', action='store_true', default=True,
                        help='Use advantage normalization')
    parser.add_argument('--reward_scale', type=float, default=1.0,
                        help='Reward scaling factor')
    parser.add_argument('--reward_clip', type=float, default=10.0,
                        help='Reward clipping threshold')

    # === Exploration Strategy Configuration  ===
    parser.add_argument('--exploration_method', type=str, default='adaptive',
                        choices=['fixed', 'adaptive', 'scheduled', 'epsilon_greedy'],
                        help='Exploration strategy type')
    parser.add_argument('--initial_exploration_rate', type=float, default=0.3,
                        help='Initial exploration rate - water resource environment needs larger initial exploration')
    parser.add_argument('--final_exploration_rate', type=float, default=0.05,
                        help='Final exploration rate')
    parser.add_argument('--exploration_decay_steps', type=int, default=int(1.5e6),
                        help='Exploration rate decay steps - gradually reduce exploration during first 1.5M steps')
    parser.add_argument('--exploration_noise_std', type=float, default=0.1,
                        help='Exploration noise standard deviation')

    # === Early Stopping and Convergence Detection ===
    parser.add_argument('--use_early_stopping', action='store_true', default=True,
                        help='Enable early stopping mechanism')
    parser.add_argument('--early_stop_patience', type=int, default=100,
                        help='Early stopping patience - stop after 50 eval intervals without improvement')
    parser.add_argument('--convergence_threshold', type=float, default=0.005,
                        help='Convergence threshold - consider converged when reward change is less than 0.5%')
    parser.add_argument('--min_improvement', type=float, default=0.001,
                        help='Minimum improvement threshold')
    parser.add_argument('--convergence_window', type=int, default=10,
                        help='Convergence detection window size')

    # === Learning Rate Scheduling Strategy ===
    parser.add_argument('--lr_scheduler', type=str, default='linear',
                        choices=['linear', 'cosine', 'exponential', 'step'],
                        help='Learning rate scheduling strategy')
    parser.add_argument('--lr_warmup_steps', type=int, default=int(1e5),
                        help='Learning rate warmup steps - warmup during first 100k steps')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # === Memory and Performance Optimization ===
    parser.add_argument('--memory_limit', type=float, default=8.0,
                        help='Memory limit (GB)')
    parser.add_argument('--use_mixed_precision', action='store_true', default=False,
                        help='Use mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Checkpoint saving interval')

    # === Multi-agent Specific Configuration ===
    parser.add_argument('--use_agent_specific_normalization', action='store_true', default=True,
                        help='Use agent-specific feature normalization')
    parser.add_argument('--agent_interaction_penalty', type=float, default=0.0,
                        help='Agent interaction penalty coefficient')
    parser.add_argument('--coordination_bonus', type=float, default=0.1,
                        help='Coordination reward coefficient')

    # === Environment-specific Advanced Configuration ===
    parser.add_argument('--water_constraint_weight', type=float, default=1.0,
                        help='Water resource constraint weight')
    parser.add_argument('--ecological_constraint_weight', type=float, default=0.8,
                        help='Ecological constraint weight')
    parser.add_argument('--stability_reward_weight', type=float, default=0.6,
                        help='Stability reward weight')

    # === Debugging and Monitoring ===
    parser.add_argument('--debug_mode', action='store_true', default=False,
                        help='Enable debug mode')
    parser.add_argument('--profile_training', action='store_true', default=False,
                        help='Performance profiling')
    parser.add_argument('--log_gradients', action='store_true', default=False,
                        help='Log gradient information')
    parser.add_argument('--log_weights', action='store_true', default=False,
                        help='Log weight information')
    parser.add_argument('--detailed_logging', action='store_true', default=True,
                        help='Detailed logging')

    # === Other Missing MAPPO Parameters ===
    parser.add_argument('--use_clipped_value_loss', action='store_true', default=True,
                        help='Use clipped value loss')
    parser.add_argument('--use_max_grad_norm', action='store_true', default=True,
                        help='Use gradient clipping')
    parser.add_argument('--stacked_frames', type=int, default=1,
                        help='Number of stacked input frames')
    parser.add_argument('--use_naive_recurrent_policy', action='store_true', default=False,
                        help='Use simple recurrent policy')

    parser.add_argument("--use_simplified_actions", type=bool, default=True,
                        help="Use simplified discrete action space")
    parser.add_argument("--simplified_action_debug", type=bool, default=False,
                        help="Enable simplified action debug output")

    return parser


def get_optimized_hyperparameters():
    """Hyperparameter dictionary"""

    return {
        # === Core Training Parameters ===
        "episode_length": 96,
        "num_env_steps": int(1300000),
        "n_rollout_threads": 16,

        # === Buffer and Sampling Configuration - New ===
        "buffer_size": 2048,
        "rollout_length": 96,
        "batch_size": 512,
        "use_rollout_active_masks": True,

        # === Learning Rate Configuration ===
        "lr": 3e-5,
        "critic_lr": 1.2e-4,
        "use_linear_lr_decay": True,
        "lr_scheduler": "linear",
        "lr_warmup_steps": int(1e5),
        "min_lr": 1e-6,

        # === PPO Configuration ===
        "ppo_epoch": 5,
        "clip_param": 0.04,
        "entropy_coef": 0.02,
        "value_loss_coef": 1.0,
        "max_grad_norm": 0.4,
        "num_mini_batch": 4,

        # === Reward Processing and Normalization  ===
        "use_reward_normalization": True,
        "use_advantage_normalization": True,
        "reward_scale": 1.0,
        "reward_clip": 10.0,

        # === Exploration Strategy Configuration  ===
        "exploration_method": "adaptive",
        "initial_exploration_rate": 0.3,
        "final_exploration_rate": 0.05,
        "exploration_decay_steps": int(1.5e6),
        "exploration_noise_std": 0.1,

        # === Early Stopping and Convergence Detection ===
        "use_early_stopping": True,
        "early_stop_patience": 100,
        "convergence_threshold": 0.005,
        "min_improvement": 0.001,
        "convergence_window": 10,

        # === Network Architecture ===
        "hidden_size": 256,
        "layer_N": 3,
        "use_feature_normalization": True,
        "use_centralized_V": True,
        "use_agent_specific_normalization": True,

        # === GAE Parameters ===
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "use_gae": True,

        # === Evaluation Settings ===
        "eval_interval": 500,
        "eval_episodes": 5,
        "use_eval": True,

        # === Multi-agent Coordination ===
        "agent_interaction_penalty": 0.0,
        "coordination_bonus": 0.1,

        # === Environment-specific Weights ===
        "water_constraint_weight": 1.0,
        "ecological_constraint_weight": 0.8,
        "stability_reward_weight": 0.6,

        # === Performance Optimization ===
        "memory_limit": 8.0,
        "gradient_accumulation_steps": 1,
        "checkpoint_interval": 100,
        "detailed_logging": True,

        # === Environment-specific ===
        "continuous_management": True,
        "num_reservoirs": 15,
        "num_plants": 6,

        # === Performance Monitoring Metric Thresholds ===
        "target_water_satisfaction": 0.95,
        "target_ecological_compliance": 0.90,
        "target_stable_zone_rate": 0.90,
        "max_flood_risk": 0.05,
    }


def print_config_summary(args):

    print("=" * 60)
    print("Water Resource Management Training Configuration")
    print("=" * 60)

    print("Environment Configuration:")
    print(f"   Episode length: {args.episode_length} hours ")
    print(f"   Number of agents: {args.num_reservoirs + args.num_plants} (Reservoirs: {args.num_reservoirs} + Plants: {args.num_plants})")

    print("\n️ Training Configuration:")
    print(f"   Total training steps: {args.num_env_steps:,}")
    print(f"   Learning rates: Actor {args.lr:.1e}, Critic {args.critic_lr:.1e}")
    print(f"   Network size: {args.hidden_size} x {args.layer_N} layers")

    print("=" * 60)


if __name__ == "__main__":
    # Example
    parser = get_config()
    args = parser.parse_args()
    print_config_summary(args)

    # Get optimized hyperparameters
    hyperparams = get_optimized_hyperparameters()
    print("\n Key Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"   {key}: {value}")