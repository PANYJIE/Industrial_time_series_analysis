import torch


def Test(feature_set_test, target_set_test, model_path, device):
    device = torch.device(device)
    model = torch.load(model_path)

    feature_set_test = feature_set_test.to(device)
    target_set_test = target_set_test.to(device)
    predict = model(feature_set_test)
    true = target_set_test[:, :, 0]


    return predict, true



