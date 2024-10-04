from model.torch_model_base import TorchModelBase
import torch.nn as nn


class Model(TorchModelBase):

    def __init__(self, num_users, num_items, num_factors, **kwargs):
        super(Model, self).__init__()

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.mu = kwargs["mu"]

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)
        nn.init.xavier_normal_(self.user_bias.weight)
        nn.init.xavier_normal_(self.item_bias.weight)

    def forward(self, user_idx, item_idx):
        embed_user = self.embed_user(user_idx) # batch_size * num_factors
        embed_item = self.embed_item(item_idx) # batch_size * num_factors
        user_bias = self.user_bias(user_idx) # batch_size * 1
        item_bias = self.item_bias(item_idx) # batch_size * 1
        output = (embed_user * embed_item).sum(axis=1) + user_bias + item_bias + self.mu # batch_size * 1
        return output