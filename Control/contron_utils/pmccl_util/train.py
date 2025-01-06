from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.test import testtt
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.util import ewc_utils
import torch

def train_on_batch(model=None, save_path='', pre_model=None, config={}, train_dataloader=None, val_dataloader=None, ewc=None):
    model_name = config['model']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=config['decay'])

    train_loss_list = []
    val_loss_list = []



    i = 0
    epoch = config['epoch']
    early_stop_win = 75

    stop_improve_count = 0
    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()


        if ewc is None:

            loss = ewc_utils.normal_train(model, optimizer, train_dataloader, model_name)
            print('Normal training')

        else:
            loss = ewc_utils.ewc_train(model, optimizer, train_dataloader, ewc, 5, pre_model, model_name)
            print('Continuous learning training')

        train_loss_list.append(loss)
        acu_loss += loss

        i += 1

        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
            i_epoch, epoch,
            acu_loss / len(dataloader), acu_loss), flush=True
        )


        val_model = model
        val_model.eval()

        if i_epoch > 5:
            if val_dataloader is not None:

                val_loss, val_result = testtt(val_model, val_dataloader, model_name)
                val_loss_list.append(val_loss)
                print(f'epoch {(i_epoch)} Loss:{val_loss}')
                if i_epoch == 6:
                    min_loss = val_loss
                else:
                    if val_loss < min_loss:
                        torch.save(val_model.state_dict(), save_path)
                        min_loss = val_loss
                        stop_improve_count = 0
                    else:
                        stop_improve_count += 1

                    if stop_improve_count >= early_stop_win:
                        break
            else:
                if acu_loss < min_loss:
                    torch.save(val_model.state_dict(), save_path)
                    min_loss = acu_loss


    # font1 = {'family': 'Arial', 'weight': 'normal', 'size': 18}
    # time_points1 = len(train_loss_list)
    # t1 = np.arange(0, time_points1)
    # time_points = len(val_loss_list)
    # t = np.arange(0, time_points)
    # train_loss_list = np.asarray(train_loss_list)
    # val_loss_list = np.asfarray(val_loss_list)
    # plt.plot(t, val_loss_list, label='val_loss', color='green')
    # plt.legend(prop=font1, loc='upper center', ncol=2)
    # plt.xlabel('Time (Second)', fontproperties='Times New Roman', fontsize=20)
    # plt.xticks(fontproperties='Arial', fontsize=16)
    # plt.ylabel('Value', fontproperties='Times New Roman', fontsize=20)
    # plt.yticks(fontproperties='Arial', fontsize=16)
    # plt.show()
    # plt.close()
    # plt.plot(t1, train_loss_list, label='train_loss', color='black')
    # plt.legend(prop=font1, loc='upper center', ncol=2)
    # plt.xlabel('Time (Second)', fontproperties='Times New Roman', fontsize=20)
    # plt.xticks(fontproperties='Arial', fontsize=16)
    # plt.ylabel('Value', fontproperties='Times New Roman', fontsize=20)
    # plt.yticks(fontproperties='Arial', fontsize=16)
    # plt.show()
    # plt.close()

    # return train_loss_list, ewc
    return torch.save(val_model.state_dict(), save_path)
