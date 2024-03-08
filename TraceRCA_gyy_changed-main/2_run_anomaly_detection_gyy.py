import pickle
import time
from pathlib import Path
import numpy as np

threshold = 1.0


def anomaly_detection_3sigma(df, result_column, useful_feature):
    """
    多度量调用异常检测

    :param df:故障发生后的数据
    :param result_column:我们使用的预测方法的结果的列名
    :param useful_feature:每次调用的有用特性(dict)
    :return :检测到异常调用的数据
    """
    indices = np.unique(df.index.values)
    for source, target in indices:
        # all features are not useful  没有找到有用的特征
        if (source, target) not in useful_feature:
            df.loc[(source, target), result_column] = 0
            continue
        # 此 (source, target) 调用的有用特征 features (list)
        features = useful_feature[(source, target)]
        empirical = df.loc[(source, target), features].values  # 取出有用的特征的值
        mean, std = [], []
        # 分别求取对应特征值列表的 平均值和标准值
        empirical_mean, empirical_std = np.mean(empirical, axis=0), np.std(empirical, axis=0)
        for idx, _ in enumerate(features):
            mean.append(empirical_mean[idx])
            std.append(np.maximum(empirical_std[idx], 0.1))
        mean = np.asarray(mean)
        std = np.asarray(std)
        predict = np.zeros(empirical.shape)
        for idx, _ in enumerate(features):
            # 判断该feature是否异常  理论上应该是使用正常标准值来判断 此处有问题
            predict[:, idx] = np.abs(empirical[:, idx] - mean[idx]) > threshold * std[idx]

        # 选取该调用的各feature严重程度的最大值,只要有一个feature是异常的，该调用就是异常的
        predict = np.max(predict, axis=1)

        df.loc[(source, target), result_column] = predict
    return df


def anomaly_detection_by_vector(df, result_column, useful_feature):
    """
    多度量调用异常检测

    :param df:故障发生后的数据
    :param result_column:我们使用的预测方法的结果的列名
    :param useful_feature:每次调用的有用特性(dict)
    :return :检测到异常调用的数据
    """
    indices = np.unique(df.index.values)
    for source, target in indices:
        # all features are not useful  没有找到有用的特征
        if (source, target) not in useful_feature:
            df.loc[(source, target), result_column] = 0
            continue
        # 此 (source, target) 调用的有用特征 features (list)
        features = useful_feature[(source, target)]
        empirical = df.loc[(source, target), features].values  # 取出有用的特征的值
        mean, std = [], []
        # 分别求取对应特征值列表的 平均值和标准值
        empirical_mean, empirical_std = np.mean(empirical, axis=0), np.std(empirical, axis=0)
        for idx, _ in enumerate(features):
            mean.append(empirical_mean[idx])
            std.append(np.maximum(empirical_std[idx], 0.1))
        mean = np.asarray(mean)
        std = np.asarray(std)
        predict = np.zeros(empirical.shape)
        for idx, _ in enumerate(features):
            # 判断该feature是否异常  理论上应该是使用正常标准值来判断 此处有问题
            predict[:, idx] = np.abs(empirical[:, idx] - mean[idx]) > threshold * std[idx]

        # 选取该调用的各feature严重程度的最大值,只要有一个feature是异常的，该调用就是异常的
        predict = np.max(predict, axis=1)

        df.loc[(source, target), result_column] = predict
    return df


def invo_anomaly_detection_main(input_file, output_file, useful_feature, main_threshold):
    """
    主要选择功能有用的特性

    :param input_file:故障发生后的数据
    :param output_file:异常调用检测的数据
    :param useful_feature:每个调用的有用特性(dict)
    :param main_threshold:用于比较异常严重程度的给定阈值
    :return :检测异常调用的数据
    """
    global threshold
    threshold = main_threshold
    with open(useful_feature, 'r') as f:
        useful_feature = eval("".join(f.readlines()))

    cache = None

    input_file = Path(input_file)

    with open(input_file, 'rb') as f:
        df = pickle.load(f)
    df = df.set_index(keys=['source', 'target'], drop=False).sort_index()
    tic = time.time()
    df = anomaly_detection_3sigma(df, 'Ours-predict', useful_feature)
    toc = time.time()
    print("algo:", "ours", "time:", toc - tic, 'invos:', len(df))
    df['predict'] = df['Ours-predict']
    with open(output_file, 'wb+') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    from train_config import dataset_path

    input_file = rf'{dataset_path}\admin-order_abort_1011_data.pkl'
    useful_feature = rf'{dataset_path}\useful_feature_2'
    output_file = rf'{dataset_path}\invo_anomaly_detection_2.pkl'
    main_threshold = 1

    invo_anomaly_detection_main(input_file=input_file, output_file=output_file,
                                useful_feature=useful_feature, main_threshold=main_threshold)


