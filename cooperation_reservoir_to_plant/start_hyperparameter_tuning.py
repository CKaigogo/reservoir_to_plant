import os
import sys
import torch
import wandb
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['WANDB_INIT_TIMEOUT'] = '300'
os.environ['WANDB_SILENT'] = 'true'

# 将项目根目录添加到Python路径
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))
if not (current_dir / "onpolicy").exists():
    sys.path.append(str(current_dir.parent))

from training_config import get_config

# 导入训练脚本
from train_water import train

def safe_wandb_init(max_retries=3, timeout=120):
    """安全的W&B初始化"""
    for attempt in range(max_retries):
        try:
            print(f"尝试初始化W&B (第 {attempt + 1}/{max_retries} 次)")

            # 使用更保守的设置
            run = wandb.init(
                settings=wandb.Settings(
                    init_timeout=timeout,
                    start_method='thread',
                    _disable_stats=True,
                    _disable_meta=True,
                    console='off'
                )
            )

            print(f" W&B初始化成功: {run.name}")
            return run

        except Exception as e:
            print(f" W&B初始化失败 (尝试 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f" 等待 {5 * (attempt + 1)} 秒后重试...")
                import time
                time.sleep(5 * (attempt + 1))
            else:
                print(" 所有W&B初始化尝试失败，将以离线模式继续")
                return None

def main():
    """修复版主训练函数"""
    print("启动超参数调优...")

    run = safe_wandb_init()

    if run is None:
        print(" W&B初始化失败，使用默认配置继续训练")
        # 创建默认配置
        parser = get_config()
        all_args = parser.parse_args([])
        all_args.exp_name = "fallback_training"
        all_args.use_wandb = False
        wandb_config = {}
    else:
        # 获取基础配置
        print("--- Loading base config from get_improved_config() ---")
        parser = get_config()
        all_args = parser.parse_args([])

        # 用W&B参数覆盖
        print("--- Overriding args with W&B sweep config ---")
        wandb_config = dict(wandb.config)

        for key, value in wandb_config.items():
            if hasattr(all_args, key):
                print(f"  - Overriding '{key}': from '{getattr(all_args, key)}' to '{value}'")
                setattr(all_args, key, value)

        all_args.exp_name = run.name
        all_args.use_wandb = True

    # 必需的环境参数
    print("--- Setting fixed, non-tuned parameters ---")
    all_args.use_fixed_connections = True
    all_args.continuous_management = True
    all_args.env_name = "WaterManagement"

    # 启用简化动作空间
    all_args.use_simplified_actions = True
    print(f"  - Set 'use_simplified_actions': True")

    # 添加新的必需参数
    if not hasattr(all_args, 'num_agents'):
        all_args.num_agents = all_args.num_reservoirs + all_args.num_plants

    print(f"  - Set 'num_agents': {all_args.num_agents}")
    print(f"  - Set 'continuous_management': {all_args.continuous_management}")

    # 设置目录和设备
    if run:
        run_dir = Path(run.dir)
    else:
        run_dir = Path.cwd() / "fallback_results"
        run_dir.mkdir(exist_ok=True)

    all_args.run_dir = run_dir
    all_args.log_dir = run_dir / "logs"
    all_args.model_dir = run_dir / "models"
    all_args.log_dir.mkdir(parents=True, exist_ok=True)
    all_args.model_dir.mkdir(parents=True, exist_ok=True)

    # 设备选择
    if torch.cuda.is_available() and all_args.cuda:
        device = torch.device("cuda:0")
        print(f" 使用GPU: {torch.cuda.get_device_name(0)}")
        # 设置CUDA优化参数
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # 清理CUDA缓存
        torch.cuda.empty_cache()
        print(f" 已清理CUDA缓存")
    else:
        device = torch.device("cpu")
        print("使用CPU")

    all_args.device = device

    print("=" * 60)
    print(f" Final Configuration for Run: {all_args.exp_name}")
    print(f"  - Device: {device}")
    print(f"  - Environment: {all_args.num_reservoirs} reservoirs, {all_args.num_plants} plants")
    print(f"  - Continuous Management: {all_args.continuous_management}")
    print(f"  - W&B Enabled: {all_args.use_wandb}")
    if wandb_config:
        print(f"  - Sweep Config: {wandb_config}")
    print("=" * 60)

    print("\n Starting training with comprehensive error handling...")


    training_success = False
    error_message = ""

    try:
        train(all_args)
        training_success = True
        print("\n Training completed successfully!")

    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        error_message = "User interruption"

    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e):
            print(f"\n CUDA错误: {e}")
            print(" 尝试减少batch_size或使用CPU训练")
            error_message = f"CUDA error: {e}"
        elif "tensor" in str(e).lower() and "device" in str(e).lower():
            print(f"\n Tensor设备错误: {e}")
            print("建议: 检查tensor设备一致性")
            error_message = f"Tensor device error: {e}"
        else:
            print(f"\n Runtime错误: {e}")
            error_message = f"Runtime error: {e}"

    except Exception as e:
        print(f"\n Training error: {e}")
        import traceback
        traceback.print_exc()
        error_message = str(e)

    finally:
        # 安全的W&B清理
        try:
            if run and all_args.use_wandb:
                if not training_success:
                    wandb.log({"training_success": False, "error_message": error_message})
                else:
                    wandb.log({"training_success": True})

                print("正在同步W&B数据...")
                run.finish()
                print("W&B数据同步完成")

        except Exception as cleanup_error:
            print(f" W&B清理时出错: {cleanup_error}")

        # 最终CUDA清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(" 已清理CUDA缓存")

        print(f"\n 训练结果: {'成功' if training_success else '失败'}")
        if not training_success:
            print(f" 错误信息: {error_message}")


if __name__ == "__main__":
    main()