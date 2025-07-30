import numpy as np
import math
import torch
from gym import spaces

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if isinstance(obs_space, spaces.Box):
        return obs_space.shape
    elif isinstance(obs_space, spaces.Dict):
        # 取第一个子空间
        first_key = list(obs_space.spaces.keys())[0]
        return obs_space.spaces[first_key].shape
    elif isinstance(obs_space, dict):
        # 取第一个智能体的空间
        first_key = list(obs_space.keys())[0]
        return obs_space[first_key].shape
    else:
        raise NotImplementedError(f"不支持的观察空间类型: {type(obs_space)}")

def get_shape_from_act_space(act_space):
    """
    从动作空间获取最大动作维度
    Args:
        act_space: 动作空间，可以是Box、Discrete、Dict、dict等类型
    Returns:
        int: 所有智能体动作空间的最大维度
    """
    if isinstance(act_space, spaces.Box):
        return act_space.shape[0]
    elif isinstance(act_space, spaces.Discrete):
        return 1
    elif isinstance(act_space, spaces.Dict):
        # 对于gym的Dict空间
        return max(get_shape_from_act_space(space) for space in act_space.spaces.values())
    elif isinstance(act_space, dict):
        # 对于多智能体环境，返回最大动作维度
        return max(get_shape_from_act_space(v) for v in act_space.values())
    else:
        raise NotImplementedError(f"不支持的动作空间类型: {type(act_space)}")

def get_shape_from_act_space_dict(act_space):
    """
    从动作空间获取形状字典
    Args:
        act_space: 动作空间，可以是Box、Discrete、Dict、dict等类型
    Returns:
        dict: 每个智能体的动作维度字典
    """
    if isinstance(act_space, spaces.Box):
        return act_space.shape[0]
    elif isinstance(act_space, spaces.Discrete):
        return act_space.n
    elif isinstance(act_space, spaces.Dict):
        # 对于gym的Dict空间
        return {k: get_shape_from_act_space_dict(v) for k, v in act_space.spaces.items()}
    elif isinstance(act_space, dict):
        # 对于多智能体环境，返回每个智能体的动作维度字典
        result = {}
        for k, v in act_space.items():
            if isinstance(v, dict):
                # 处理嵌套字典结构
                result[k] = {sub_k: get_shape_from_act_space_dict(sub_v) for sub_k, sub_v in v.items()}
            else:
                result[k] = get_shape_from_act_space_dict(v)
        return result
    else:
        raise NotImplementedError(f"不支持的动作空间类型: {type(act_space)}")

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c
