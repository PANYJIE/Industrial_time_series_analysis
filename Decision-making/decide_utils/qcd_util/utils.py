import math


'''functions文件中调用的函数'''

'''计算MIC_ND'''
def MIC_matrix(df):  # 计算MIC矩阵
    from sklearn import metrics
    import pandas as pd
    import numpy as np
    from minepy import MINE
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    number = df.columns.size  # 获取df的列数
    List = []
    Name = []
    for n in range(number):
        Name.append(df.columns[n])  # 获取dataframe的索引
    for i in range(number):
        A = []
        X = df[df.columns[i]]  # df.columns[i]获取对应列的索引，df['索引']获取对应列的数值
        for j in range(number):
            if(i == j):
                A.append(1.0)
            else:
                Y = df[df.columns[j]]
                mine.compute_score(X, Y)
                A.append(mine.mic())  # 计算MIC信息
        List.append(A)  # List是列表格式
    # print(pd.DataFrame(List, index=Name, columns=Name))
    return pd.DataFrame(List).values

def ND(mat, beta=0.99, alpha=1, control=0):
    '''
    This is a python implementation/translation of network deconvolution by MIT-KELLIS LAB


     LICENSE: MIT-KELLIS LAB


     AUTHORS:
        Algorithm was programmed by Soheil Feizi.
        Paper authors are S. Feizi, D. Marbach,  M. M?©dard and M. Kellis
    Python implementation: Gideon Rosenthal

    REFERENCES:
       For more details, see the following paper:
        Network Deconvolution as a General Method to Distinguish
        Direct Dependencies over Networks
        By: Soheil Feizi, Daniel Marbach,  Muriel Médard and Manolis Kellis
        Nature Biotechnology

    --------------------------------------------------------------------------
     ND.m: network deconvolution
    --------------------------------------------------------------------------

    DESCRIPTION:

     USAGE:
        mat_nd = ND(mat)
        mat_nd = ND(mat,beta)
        mat_nd = ND(mat,beta,alpha,control)


     INPUT ARGUMENTS:
     mat           Input matrix, if it is a square matrix, the program assumes
                   it is a relevance matrix where mat(i,j) represents the similarity content
                   between nodes i and j. Elements of matrix should be
                   non-negative.
     optional parameters:
     beta          Scaling parameter, the program maps the largest absolute eigenvalue
                   of the direct dependency matrix to beta. It should be
                   between 0 and 1.
     alpha         fraction of edges of the observed dependency matrix to be kept in
                   deconvolution process.
     control       if 0, displaying direct weights for observed
                   interactions, if 1, displaying direct weights for both observed and
                   non-observed interactions.

     OUTPUT ARGUMENTS:

     mat_nd        Output deconvolved matrix (direct dependency matrix). Its components
                   represent direct edge weights of observed interactions.
                   Choosing top direct interactions (a cut-off) depends on the application and
                   is not implemented in this code.

     To apply ND on regulatory networks, follow steps explained in Supplementary notes
     1.4.1 and 2.1 and 2.3 of the paper.
     In this implementation, input matrices are made symmetric.

    **************************************************************************
     loading scaling and thresholding parameters
    '''
    import scipy.stats.mstats as stat
    from numpy import linalg as LA
    import numpy as np

    if beta >= 1 or beta <= 0:
        print ('error: beta should be in (0,1)')

    if alpha > 1 or alpha <= 0:
        print ('error: alpha should be in (0,1)')

    '''
    ***********************************
     Processing the inut matrix
     diagonal values are filtered
    '''

    n = mat.shape[0]
    np.fill_diagonal(mat, 0)

    '''
    Thresholding the input matrix
    '''
    y = stat.mquantiles(mat[:], prob=[1 - alpha])
    th = mat >= y
    mat_th = mat * th

    '''
    making the matrix symetric if already not
    '''
    mat_th = (mat_th + mat_th.T) / 2

    '''
    ***********************************
    eigen decomposition
    '''
    # print ('Decomposition and deconvolution...')

    Dv, U = LA.eigh(mat_th)
    D = np.diag((Dv))
    lam_n = np.abs(np.min(np.min(np.diag(D)), 0))
    lam_p = np.abs(np.max(np.max(np.diag(D)), 0))

    m1 = lam_p * (1 - beta) / beta
    m2 = lam_n * (1 + beta) / beta
    m = max(m1, m2)

    # network deconvolution
    for i in range(D.shape[0]):
        D[i, i] = (D[i, i]) / (m + D[i, i])

    mat_new1 = np.dot(U, np.dot(D, LA.inv(U)))

    '''

    ***********************************
     displying direct weights
    '''
    if control == 0:
        ind_edges = (mat_th > 0) * 1.0
        ind_nonedges = (mat_th == 0) * 1.0
        m1 = np.max(np.max(mat * ind_nonedges))
        m2 = np.min(np.min(mat_new1))
        mat_new2 = (mat_new1 + np.max(m1 - m2, 0)) * ind_edges + (mat * ind_nonedges)
    else:
        m2 = np.min(np.min(mat_new1))
        mat_new2 = (mat_new1 + np.max(-m2, 0))

    '''
    ***********************************
     linearly mapping the deconvolved matrix to be between 0 and 1
    '''
    m1 = np.min(np.min(mat_new2))
    m2 = np.max(np.max(mat_new2))
    mat_nd = (mat_new2 - m1) / (m2 - m1)

    return mat_nd

'''计算Improve_IKS'''
def gDegree(G):
    """
    将G.degree()的返回值变为字典
    """
    node_degrees_dict = {}
    for i in G.degree():
        node_degrees_dict[i[0]]=i[1]
    return node_degrees_dict.copy()

def kshell(G):
    """
    kshell(G)计算k-shell值
    """
    graph = G.copy()
    importance_dict = {}
    ks = 1
    while graph.nodes():
        temp = []
        node_degrees_dict = gDegree(graph)
        kks = min(node_degrees_dict.values())
        while True:
            for k, v in node_degrees_dict.items():
                if v == kks:
                    temp.append(k)
                    graph.remove_node(k)
                    node_degrees_dict = gDegree(graph)
            if kks not in node_degrees_dict.values():
                break
        importance_dict[ks] = temp
        ks += 1
    return importance_dict

def sumD(G):
    """
    计算G中度的和
    """
    G_degrees = gDegree(G)
    sum = 0
    for v in G_degrees.values():
        sum += v
    return sum
def get_key_from_value(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key
    return None  # 如果没有找到对应值的键，返回None
def getNodeImportIndex(G):
    """
    计算节点的重要性指数
    """
    sum = sumD(G)
    I = {}
    G_degrees = gDegree(G)
    for k,v in G_degrees.items():
        I[k] = v/sum
    return I
def ks(G,k):
    """
    ks(G) 计算出G中节点k的ks值
    """
    ks_value = kshell(G)
    for item in ks_value.values():
        if k in item:
            return get_key_from_value(ks_value, item)
def Entropy(G):
    """
    Entropy(G) 计算出G中所有节点的熵乘以ks(G,k)
    I 为重要性
    e 为节点的熵sum += I[i]*math.log(I[i])
    """
    I = getNodeImportIndex(G)
    e = {}
    for k,v in I.items():
        sum = 0
        for i in G.neighbors(k):
            sum += I[i]*math.log(I[i])*ks(G, k)
        sum = -sum
        e[k] = sum
    return e
def kshellEntropy(G):
    """
    kshellEntropy(G) 是计算所有壳层下，所有节点的熵值
    例：
    {28: {'1430': 0.3787255719932099,
          '646': 0.3754626894107377,
          '1431': 0.3787255719932099,
          '1432': 0.3787255719932099,
          '1433': 0.3754626894107377
          ....
    ks is a dict 显示每个壳中的节点
    e 计算了算有节点的熵
    """
    ks = kshell(G)
    e = Entropy(G)
    ksES = {}
    ksI = max(ks.keys())
    while ksI > 0:
        ksE = {}
        for i in ks[ksI]:
            ksE[i] = e[i]
        ksES[ksI] = ksE
        ksI -= 1
    return ksES

'''计算'''