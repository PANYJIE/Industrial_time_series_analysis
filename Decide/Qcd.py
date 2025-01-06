from Industrial_time_series_analysis.Decide.decide_utils.qcd_util.functions import data_preprocess, MIC_ND_process, improve_IKS


'''
质量数据因果分析算法

说明：

1.注意！！！需要额外在电脑中下载安装graphviz才能正常正常显示图片！网址：https://graphviz.org/download/
2.输入数据格式：数据预处理后的格式见示例文件 ./frr.xlsx 
            数据预处理时，取出需要分析的质量因素数据，集合放到一个excel文件中。对于字符串数据，需要手工预处理数据
            使得相同种类的因素的对应字符串统一。如对于示例文件中 improve_method 这一项，在检查所有数据后可分为
            ”增加检测“，“工艺更正”，“加强培训”等等。

输入：
    data_path:原始数据路径
    threshold:网络去卷积后的过滤阈值，小于此阈值的节点之间连接将被舍去。直观的来说，阈值越低，
              过滤后的剩余的连接越多，网络越复杂，反之网络越简单。默认值为0.55。
    use_node_num:使用多少个节点进行网络分析，也就是将节点按照重要性排序后，取最重要的前多少个
              计算贝叶斯网络。默认值为24。如果此值较大，可能导致计算时间大幅度延长。
    use_node_list:使用自定义的节点列表，如[3, 4, 5, 6, 7, 8]。如果不使用就不输入。
    pic_name:贝叶斯网络图片保存的名称，默认为sample.png。
    
输出：贝叶斯网络图片，节点重要性排序结果
'''
def fit(df_data, threshold=0.55, use_node_num=24, defined_node_list=False, pic_name='sample.png'):
    # 对数据预处理。取出excel中每一列数据，全部转换为字符串类型，再调用LabelEncoder编码为数字
    df_data = data_preprocess(df_data)
    data = df_data.values
    # 计算MIC矩阵，并对矩阵进行网络去卷积，最后阈值过滤
    adj_matrix = MIC_ND_process(df_data, threshold)
    # 计算节点重要性，并返回节点重要性列表
    node_weigth_list = improve_IKS(adj_matrix)
    use_node_list = node_weigth_list[0: use_node_num]
    # 是否使用自定义列表
    if defined_node_list:
        use_node_list = defined_node_list
    # 构建贝叶斯网络
    # 根据节点重要性列取出原数据中需要的列
    b_data = data[:, use_node_list]
    b_labels = list(df_data.columns[use_node_list])

    return node_weigth_list, b_data, b_labels








