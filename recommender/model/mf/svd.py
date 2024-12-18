from model.torch_model_base import TorchModelBase
import torch.nn as nn


class Model(TorchModelBase):

    def __init__(self, num_users, num_items, num_factors, **kwargs):
        super(Model, self).__init__()

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    def forward(self, user_idx, item_idx):
        embed_user = self.embed_user(user_idx) # batch_size * num_factors
        embed_item = self.embed_item(item_idx) # batch_size * num_factors
        output = (embed_user * embed_item).sum(axis=1) # batch_size * 1
        return output