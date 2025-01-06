import pandas as pd
from sklearn.utils import resample

def fit(data, target_column, rate):
    """
    :param data: 需要处理的数据集
    :param target_column: 标签的列名
    :return: 平衡后的数据集
    """
    # 获取每个类别的样本数量
    class_counts = data[target_column].value_counts()
    # 计算每个类别的权重
    total_samples = len(data)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    w = sum(class_weights.values())
    # 新数据集初始化
    balanced_data = pd.DataFrame()

    for cls, count in class_counts.items():
        ni_prime = max(int((count * class_weights[cls] / w) * rate), 2)

        if ni_prime <= count:
            # 下采样
            undersampled_data = resample(
                data[data[target_column] == cls],
                replace=False,
                n_samples=ni_prime,
                random_state=42
            )
            balanced_data = pd.concat([balanced_data, undersampled_data], ignore_index=True)
        else:
            # 上采样
            oversampled_data = resample(
                data[data[target_column] == cls],
                replace=True,
                n_samples=ni_prime - count,
                random_state=42
            )
            balanced_data = pd.concat([balanced_data, data[data[target_column] == cls], oversampled_data],
                                      ignore_index=True)

    return balanced_data
