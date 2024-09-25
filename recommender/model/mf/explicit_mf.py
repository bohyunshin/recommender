from model.torch_model_base import TorchModelBase
import torch.nn as nn


class MatrixFactorization(TorchModelBase):

    def __init__(self, num_users, num_items, num_factors):
        super(MatrixFactorization, self).__init__()

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)
        # predict_size = num_factors
        # self.predict_layer = torch.ones(predict_size, 1).cuda()
        # self._init_weight_()

    def forward(self, user_idx, item_idx):
        embed_user = self.embed_user(user_idx) # batch_size * num_factors
        embed_item = self.embed_item(item_idx) # batch_size * num_factors
        output = (embed_user * embed_item).sum(axis=1) # batch_size * 1
        return output