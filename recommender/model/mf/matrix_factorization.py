import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):

    def __init__(self, num_users, num_items, num_factors):
        super(MatrixFactorization, self).__init__()

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)
        # predict_size = num_factors
        # self.predict_layer = torch.ones(predict_size, 1).cuda()
        # self._init_weight_()

    def forward(self, user_idx, item_idx):
        embed_user = self.embed_user(user_idx)
        embed_item = self.embed_item(item_idx)
        output = torch.matmul(embed_user, embed_item)
        return output


if __name__ == "__main__":
    mf = MatrixFactorization(1000, 3000, 200)
    print("hi")