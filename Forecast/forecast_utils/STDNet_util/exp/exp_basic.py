import os
import torch
# from Industrial_time_series_analysis.Forecast.forecast_utils.STDNet_util.models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
#     Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, NBEATS, MSNet, MSNet2, MSNet3, MSNet4, MSNet5, MSNet6, MSNet7, MSNet8, \
#           MSNet9, MSNet10, MSNet9_ana, MSNet11, MSNet12, MSNet9_att, MSNet9_emd, MSNet9_2dconv, MSNet9_multi_re

from Industrial_time_series_analysis.Forecast.forecast_utils.STDNet_util.models import  MSNet9,stdnet1

from Industrial_time_series_analysis.Forecast.forecast_utils.STDNet_util.models.stdnet import stdnet

# from Industrial_time_series_analysis.Forecast.forecast_utils.STDNet_util.models.stdnet.stdnet_abb import stdnet_att

# from Industrial_time_series_analysis.Forecast.forecast_utils.STDNet_util.models.stdnet.stdnet_abb import stdnet_cnn


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'STDNet' : stdnet1
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
