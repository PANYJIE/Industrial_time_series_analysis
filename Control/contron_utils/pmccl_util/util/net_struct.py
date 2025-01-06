

def get_feature_map(list_data):
    feature_list = list_data



    return feature_list


def get_fc_graph_struc(path):
    feature_file = open(f'{path}/list.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)
    
    return struc_map

def get_prior_graph_struc(list_data, adjacency_matrix):

    adjacency_matrix = adjacency_matrix.values.tolist()



    struc_map = {}
    feature_list = list_data


    for ft in feature_list:
        ft_index = feature_list.index(ft)
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            other_ft_index = feature_list.index(other_ft)
            if other_ft is not ft and adjacency_matrix[ft_index][other_ft_index] == 1:
                struc_map[ft].append(other_ft)


    
    return struc_map


 