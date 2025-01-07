from Industrial_time_series_analysis.Decide.decide_utils.qcd_util.functions import data_preprocess, MIC_ND_process, improve_IKS

def fit(df_data, threshold=0.55, use_node_num=24, defined_node_list=False, pic_name='sample.png'):
    df_data = data_preprocess(df_data)
    data = df_data.values
    adj_matrix = MIC_ND_process(df_data, threshold)
    node_weigth_list = improve_IKS(adj_matrix)
    use_node_list = node_weigth_list[0: use_node_num]
    if defined_node_list:
        use_node_list = defined_node_list
    b_data = data[:, use_node_list]
    b_labels = list(df_data.columns[use_node_list])

    return node_weigth_list, b_data, b_labels








