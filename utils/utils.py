import numpy as np
import torch
import random

import time
from datetime import timedelta


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_onehot_to_onelabel(onehot):
    assert sum(onehot) <= 1
    label = 0
    for idx, one in enumerate(onehot):
        if one == 1:
            label = idx + 1

    return label
