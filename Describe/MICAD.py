import networkx as nx


def build_assemblies_dict(df):
    assemblies_dict = {}
    for index, row in df.iterrows():
        key, value = row[0], row[1]
        if key in assemblies_dict:
            assemblies_dict[key].append(value)
        else:
            assemblies_dict[key] = [value]
    return assemblies_dict


def build_network(df, assemblies_dict):
    G = nx.Graph()
    unique_parts = sorted(df['组件'].unique().tolist() + df['连接组件'].unique().tolist())
    part_to_id = {part: idx for idx, part in enumerate(unique_parts)}
    id_to_part = {idx: part for part, idx in part_to_id.items()}

    for parent, children in assemblies_dict.items():
        parent_id = part_to_id[parent]
        G.add_node(parent_id, label=parent)
        for child in children:
            child_id = part_to_id[child]
            G.add_node(child_id, label=child)
            G.add_edge(parent_id, child_id, weight=1)

    for node in G.nodes():
        edges = G.edges(node, data=True)
        degree = G.degree(node)
        if degree > 1:
            for _, to, data in edges:
                data['weight'] = 1 / degree

    return G, part_to_id, id_to_part

def compute_centralities(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    return degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality


def compute_network_metrics(G):
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_shortest_path_length = nx.average_shortest_path_length(G)
    else:
        diameter = "The network is not fully connected to calculate the diameter"
        avg_shortest_path_length = "The network is not fully connected and the average shortest path length cannot be calculated"

    density = nx.density(G)
    return diameter, avg_shortest_path_length, density


def get_top_n_centralities(centrality_dict, n=10):
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]


def fit(df):
    assemblies_dict = build_assemblies_dict(df)
    G, part_to_id, id_to_part = build_network(df, assemblies_dict)

    degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality = compute_centralities(G)

    top_degree_centrality = get_top_n_centralities(degree_centrality)
    top_betweenness_centrality = get_top_n_centralities(betweenness_centrality)
    top_closeness_centrality = get_top_n_centralities(closeness_centrality)
    top_eigenvector_centrality = get_top_n_centralities(eigenvector_centrality)
    return G, id_to_part, top_degree_centrality, top_betweenness_centrality, top_closeness_centrality, top_eigenvector_centrality



