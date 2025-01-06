import torch.nn as nn
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.util.util_time import *
from .util.util_time import *
from .util.env import *

def testtt(model, dataloader, model_name='GDN'):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0

    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad():
            if model_name == 'GDN':
                predicted = model(x, edge_index).float().to(device)

            else:
                predicted = model(x).float().to(device)
            # loss = loss_func(predicted[:,7], y[:,7])
            loss = loss_func(predicted, y)

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])


            if len(test_predicted_list) <= 0:
                test_predicted_list = predicted
                test_ground_list = y
                test_labels_list = labels

            else:
                test_predicted_list = torch.cat((test_predicted_list, predicted), dim=0)
                test_ground_list = torch.cat((test_ground_list, y), dim=0)
                test_labels_list = torch.cat((test_labels_list, labels), dim=0)


        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    test_predicted_list = test_predicted_list.tolist()
    test_ground_list = test_ground_list.tolist()
    test_labels_list = test_labels_list.tolist()


    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]
