import numpy as np
import torch
from copy import deepcopy

def oil_data_prepare(df_data, data_mean_save_path, data_var_save_path, data_num_per_sample, data_predict_step):

    data = np.array(df_data)
    data = data[:, 1:]
    data = np.flipud(data)
    all_data = deepcopy(data)
    print((all_data.shape))

    num_per_sample = data_num_per_sample
    roll_skip_step = 4

    all_samples = []
    sample_idx = 0
    while(1):
        if sample_idx % 1000 == 0:
            print('idx', sample_idx)
            print('sample len', len(all_samples))
        if sample_idx > 40000:
            break
        temp_sample = []
        data = all_data[sample_idx:sample_idx+num_per_sample]
        data = np.array(data)
        for j, n in zip(data, range(num_per_sample)):
            temp_sample.append(j)
        all_samples.append(temp_sample)
        sample_idx = sample_idx + roll_skip_step
    all_samples = np.array(all_samples[0:10000])
    all_samples = all_samples.astype('float32')
    print('all sample shape:', all_samples.shape, all_samples.dtype)


    all_data = all_samples
    dim0 = all_data.shape[0]
    dim1 = all_data.shape[1]
    dim2 = all_data.shape[2]

    all_data = np.reshape(all_data, (dim0*dim1, dim2))
    d_max = np.max(all_data, axis=0)
    d_min = np.min(all_data, axis=0)
    d_mean = np.mean(all_data, axis=0)
    d_var = np.std(all_data, axis=0)
    np.save(data_mean_save_path, d_mean)
    np.save(data_var_save_path, d_var)
    # nor = (all_data - d_min) / (d_max - d_min)
    nor = (all_data - d_mean) / (d_var + 1e-8)
    all_data = nor.reshape(dim0, dim1, dim2)
    print('data shape:', all_data.shape, all_data.dtype)

    num_per_sample = data_num_per_sample
    predict_step = data_predict_step


    k_flod = 10
    train_set_ratio = 0.8
    val_set_ratio = 0.1

    train_set_fea, train_set_tar, val_set_fea, val_set_tar, test_set_fea, test_set_tar = [], [], [], [], [], []
    sign = deepcopy(k_flod)

    def split_and_save_fea_tar(data, fea_set, tar_set):
        fea = data[:num_per_sample-predict_step]
        tar = data[num_per_sample-predict_step:]
        fea_set.append(fea)
        tar_set.append(tar)
    for each_data in all_data:
        if sign == 0:
           sign = deepcopy(k_flod)

        if k_flod - k_flod*train_set_ratio <sign<= k_flod:
            split_and_save_fea_tar(each_data, train_set_fea, train_set_tar)

        if k_flod - k_flod*train_set_ratio - k_flod*val_set_ratio <sign<= k_flod - k_flod*train_set_ratio:
            split_and_save_fea_tar(each_data, val_set_fea, val_set_tar)
        elif sign <= k_flod - k_flod * train_set_ratio - k_flod * val_set_ratio:
            split_and_save_fea_tar(each_data, test_set_fea, test_set_tar)
        sign -= 1

    train_set_fea = torch.tensor(train_set_fea, dtype=torch.float32)
    train_set_tar = torch.tensor(train_set_tar, dtype=torch.float32)


    val_set_fea = torch.tensor(val_set_fea, dtype=torch.float32)
    val_set_tar = torch.tensor(val_set_tar, dtype=torch.float32)

    test_set_fea = torch.tensor(test_set_fea, dtype=torch.float32)
    test_set_tar = torch.tensor(test_set_tar, dtype=torch.float32)

    print('train set shape')
    print(np.array(train_set_fea).shape, np.array(train_set_tar).shape)
    print('val set shape')
    print(np.array(val_set_fea).shape, np.array(val_set_tar).shape)
    print('test set shape')
    print(np.array(test_set_fea).shape, np.array(test_set_tar).shape)
    return train_set_fea, train_set_tar, val_set_fea, val_set_tar, test_set_fea, test_set_tar