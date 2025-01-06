import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .utils import MIC_matrix, ND, kshellEntropy
import networkx as nx




'''main文件中调用的函数'''

# 数据预处理函数。取出每一列数据，然后转换为字符串类型，再进行LabelEncoder编码
# 输入：数据路径:str
# 输出：处理后的数据:DataFrame
def data_preprocess(df_data):

    df = df_data.fillna(0)

    for item in df.columns.tolist():
        df[item] = df[item].astype('str')
        le = LabelEncoder()
        le.fit(df[item])
        df[item] = le.transform(df[item])
    return df

# 计算MIC矩阵，并对矩阵进行网络去卷积
# 输入：预处理后数据:DataFrame，过滤阈值:float
# 输出：关系矩阵:array
def MIC_ND_process(df, threshold):
    MIC_result = MIC_matrix(df)
    MIC_ND_result = ND(MIC_result)
    MIC_ND_result[MIC_ND_result < threshold] = 0
    return MIC_ND_result

# 根据改进IKS算法计算节点重要性排序
# 输入：关系矩阵：array
# 输出：节点重要性排序：List
def improve_IKS(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    result_dict = kshellEntropy(G)
    sorted_dict_by_key = dict(sorted(result_dict.items()))
    # 逐行输出字典的键值对
    # print('节点重要性信息表，每一行含义如下：\n'
    #       '(k shell层数)：{(节点索引1, 节点重要性1)，(节点索引2, 节点重要性2)...}')
    # for key, value in sorted_dict_by_key.items():
    #     print(f"{key}: {value}")

    # 整理dict格式
    list_dict = []
    split_list_dict = [item for item in sorted_dict_by_key.values()]
    for each in split_list_dict:
        if len(each) > 1:
            list_of_dicts = [{key: value} for key, value in each.items()]
            list_dict += list_of_dicts
        else:
            list_dict.append(each)

    # 根据字典中的节点重要性进行排序
    sorted_data = sorted(list_dict, key=lambda x: list(x.values())[0], reverse=True)
    # 将节点索引按照重要性从大到小排序
    sorted_indices = [list(d.keys())[0] for d in sorted_data]

    return sorted_indices

