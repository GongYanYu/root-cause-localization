"""run_selecting_features.py


为每个故障选择有用的特征
========================================
我们通过测试一个特性是否有用来确定它是否有用正常和异常调用的分布相对于故障发生后的变化。
特征候选集
---------------------
在微服务系统中，有各种各样的度量。在火车票数据中，我们使用每次调用的延迟和HTTP状态，以及CPU使用率，
内存使用、网络接收/发送吞吐量和磁盘每个微服务的读写吞吐量作为特性跟踪异常检测。
请注意，我们只考虑同一微服务的历史调用对，因为基于相同特性的底层发行版可以不同的微服务对差异很大。



参数
1. Input_file:故障发生后的数据(pkl)
2. 历史:同一微服务对的所有历史调用
1)在最后一个槽位，2)在最后一个周期的同一槽位(pkl)
3.Output_file:每次调用的有用特性(dict)
4. fish_threshold:一个给定的阈值，用于测试调用的特性是否有用

"""

import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from train_config import FEATURE_NAMES

DEBUG = False  # very slow


def stderr_criteria(empirical, reference, threshold):
    """
    测试调用的特性是否有用

    :param empirical:故障发生后的数据
    :param reference:故障发生前的正常数据(包含双阶段)
    :param threshold:一个给定的阈值，用于测试调用的特性是否有用
    :返回:bool类型
    """
    empirical, reference = np.array(empirical), np.array(reference)
    # 正常数据的平均值  正常数据的标准差
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    historical_std = np.maximum(historical_std, historical_mean * 0.01 + 0.01)
    ref_ratio = np.mean(np.abs(reference - historical_mean)) / historical_std
    emp_ratio = np.mean(np.abs(empirical - historical_mean)) / historical_std
    return np.abs(emp_ratio - ref_ratio) > threshold * ref_ratio + 1.0


def selecting_feature_main(input_file: str, output_file: str, history: str, fisher_threshold):
    """
    选择功能有用的特性

    :param input_file:故障发生后的数据(pkl)
    :param output_file:每次调用的有用特性(dict)
    :param history:故障发生前的正常数据(包含两个阶段)
    :param fisher_threshold:一个给定的阈值，用于测试调用的特性是否有用
    :return:
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    with open(history, 'rb') as f:
        history = pickle.load(f)
    with open(str(input_file), 'rb') as f:
        df = pickle.load(f)
    # 设置索引并排序
    df = df.set_index(keys=['source', 'target'], drop=True).sort_index()
    # 将参数转换为数字类型
    df['http_status'] = pd.to_numeric(df['http_status'])
    history['http_status'] = pd.to_numeric(history['http_status'])
    history = history.set_index(keys=['source', 'target'], drop=True).sort_index()
    df_t = df.index.values
    df_t = np.unique(df_t)
    # 取交集 获取相同元素数据 即获取相同调用路径数据
    indices = np.intersect1d(df_t, np.unique(history.index.values))
    # 设置 null值默认返回值为list
    useful_features_dict = defaultdict(list)
    if DEBUG:
        plot_dir = output_file.parent / 'selecting_feature.debug'
        plot_dir.mkdir(exist_ok=True)
    # 笛卡尔积 方便遍历的 可以不这样写
    for_data = product(indices, FEATURE_NAMES)
    for (source, target), feature in tqdm(for_data):
        # 获取 source, target 索引下feature列所有数据
        empirical = np.sort(df.loc[(source, target), feature].values)
        reference = np.sort(history.loc[(source, target), feature].values)
        # 进行判断 测试调用的特性是否有用
        fisher = stderr_criteria(empirical, reference, fisher_threshold)
        if fisher:
            # 对于此调用(source, target)来说 feature特征有用 加入判断逻辑
            useful_features_dict[(source, target)].append(feature)
    with open(output_file, 'w+') as f:
        print(dict(useful_features_dict), file=f)


if __name__ == '__main__':
    dataset_path = rf'../datasets/A'
    input_file = rf'{dataset_path}\uninjection\admin-order_abort_1011_data.pkl'
    history = rf'{dataset_path}\uninjection\pkl_3_data.pkl'
    output_file = rf'{dataset_path}\uninjection\useful_feature_2'
    # 给定的阈值，用于测试调用的特性是否有用
    fisher_threshold = 1
    selecting_feature_main(input_file=input_file, output_file=output_file, history=history,
                           fisher_threshold=fisher_threshold)
