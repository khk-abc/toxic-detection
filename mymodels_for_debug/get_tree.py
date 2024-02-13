import torch
import torch.nn as nn


class TreeGetter(nn.Module):

    def __init__(self, kernel_size=None, stride=None,
                 padding=None, in_channels=None,
                 out_channels=None):
        super(TreeGetter, self).__init__()

        self.kernel_size_1 = 3
        self.padding_1 = 1
        self.stride_1 = 1
        self.kernel_size_2 = 3
        self.padding_2 = 1
        self.stride_2 = 2

        self.conv = nn.Sequential(
            nn.Conv1d(kernel_size=(self.kernel_size_1,), stride=(self.stride_1,), padding=(self.padding_1,), in_channels=768,
                      out_channels=768),
            nn.GELU(),
            # nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(kernel_size=(self.kernel_size_2,), stride=(self.stride_2,), padding=(self.padding_2,), in_channels=768,
                      out_channels=768),
        )

        # # v2
        # self.conv = nn.Sequential(
        #     nn.Conv1d(kernel_size=(self.kernel_size,), stride=(self.stride,), padding=self.padding, in_channels=768,
        #               out_channels=768),
        #     nn.GELU(),
        #     # nn.Tanh(),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(kernel_size=(self.kernel_size,), stride=(self.stride,), padding=self.padding, in_channels=768,
        #               out_channels=768),
        #     nn.GELU(),
        # )



    def init_weight(self):
        print('init weights!')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')


    def _get_out_size(self,in_dim):
        # 两次计算对应两次卷积
        in_dim = (in_dim + 2 * self.padding_1 - self.kernel_size_1) // self.stride_1 + 1
        in_dim = (in_dim + 2 * self.padding_2 - self.kernel_size_2) // self.stride_2 + 1

        return in_dim

    def _get_one_level(self, sequence_feas, lengths):
        """
        :param sequence_feas: bz, len, dim
        :param lengths:
        :return:
        """
        bz,padded_len,dim = sequence_feas.shape
        # print(bz,padded_len,dim)

        sequence_feas = sequence_feas.permute(0, 2, 1)  # to shape [bz,dim,len]
        current_level = []
        new_lengths = []
        pad_masks = []
        for b in range(bz):
            # raw conv version
            real_feas = self.conv(sequence_feas[b][:, :lengths[b]].unsqueeze(dim=0)).squeeze(0)  # [dim,len]

            padded_out_size = self._get_out_size(in_dim=sequence_feas.shape[-1])

            # print(padded_out_size)
            # print(real_feas.shape)

            current_level.append(
                torch.cat([real_feas,
                           torch.zeros((dim, padded_out_size - real_feas.shape[-1])).to(real_feas.device)],dim=-1),
            )
            new_lengths.append(real_feas.shape[-1])

            pad_mask = torch.cat([torch.ones(real_feas.shape[-1]),
                                          torch.zeros(padded_out_size - real_feas.shape[-1])],
                                         dim=-1)
            pad_masks.append(pad_mask.to(real_feas.device))

            assert pad_mask.sum() == new_lengths[-1]


            # if lengths[b]>4:  # 大于4可以进行卷积
            #     real_feas = self.conv(sequence_feas[b][:, :lengths[b]].unsqueeze(dim=0)).squeeze(0)  # [dim,len]
            #
            #     padded_out_size = self._get_out_size(in_dim=sequence_feas.shape[-1])
            #
            #     # print(padded_out_size)
            #     # print(real_feas.shape)
            #
            #     current_level.append(
            #         torch.cat([real_feas,
            #                    torch.zeros((dim, padded_out_size - real_feas.shape[-1])).to(real_feas.device)],dim=-1),
            #     )
            #     new_lengths.append(real_feas.shape[-1])
            #
            #     pad_mask = torch.cat([torch.ones(real_feas.shape[-1]),
            #                                   torch.zeros(padded_out_size - real_feas.shape[-1])],
            #                                  dim=-1)
            #     pad_masks.append(pad_mask.to(real_feas.device))
            #
            #     assert pad_mask.sum() == new_lengths[-1]
            # elif lengths[b]>1:  # 小于等于4但是大于1，求平均
            #     padded_out_size = self._get_out_size(in_dim=sequence_feas.shape[-1])
            #
            #     real_feas = (torch.sum(sequence_feas[b][:, :lengths[b]], dim=1) / lengths[b]).unsqueeze(dim=-1)
            #     if padded_out_size>1:
            #         pad_mask = torch.cat([torch.ones(1),torch.zeros(padded_out_size - 1)],dim=-1)
            #         current_level.append(torch.cat([
            #             real_feas,
            #             torch.zeros((dim, padded_out_size - real_feas.shape[-1])).to(sequence_feas.device)], dim=-1),
            #         )
            #     else:
            #         pad_mask = torch.ones(1)
            #         current_level.append(real_feas)
            #
            #
            #     pad_masks.append(pad_mask.to(real_feas.device))
            #     new_lengths.append(1)
            #
            # elif lengths[b]==1:  # 等于1，都用 0 填充
            #     padded_out_size = self._get_out_size(in_dim=sequence_feas.shape[-1])
            #     current_level.append(torch.zeros((dim, padded_out_size)).to(sequence_feas.device))
            #     pad_masks.append(torch.zeros(padded_out_size).to(sequence_feas.device))
            #     new_lengths.append(0)


        current_level = torch.stack(current_level, dim=0)  # bz,dim,len
        pad_masks = torch.stack(pad_masks, dim=0)

        return current_level.permute(0, 2, 1), new_lengths, pad_masks

    def forward(self, sequence_feas, lengths):
        tree = []

        while all([leng>1 for leng in lengths]):
        # while all([leng>4 for leng in lengths]):
        # while any([leng>4 for leng in lengths]):
            # print('get in!')
            sequence_feas,lengths,pad_masks = self._get_one_level(sequence_feas,lengths=lengths)
            # print(sequence_feas.shape)
            # print(lengths)
            tree.append([sequence_feas,pad_masks])

        tree[-1][0] = (torch.sum(tree[-1][0], dim=1) / torch.tensor(lengths).to(sequence_feas.device).reshape(len(lengths), 1)).unsqueeze(dim=1)
        tree[-1][1] = torch.ones(tree[-1][0].shape[:-1]).to(sequence_feas.device)

        return tree


if __name__ == '__main__':
    model = TreeGetter()

    sequence_feas = torch.randn(2, 160, 768)
    lengths = [35, 67,]
    # sequence_feas = torch.randn(1, 84, 768)
    # lengths = [49, ]

    # trees=[]
    # for b in range(sequence_feas.shape[0]):
    #     print("==================================")
    #     tree = model(sequence_feas[b].unsqueeze(0),lengths=[lengths[b],])
    #     trees.append(tree)
    #     print("==================================")

    # for tree in trees:
    #     # print(i[0].shape)
    #     print("==================================")
    #     print('depth: ',len(tree))
    #     for i in tree:
    #         print(i[1].sum(dim=-1))
    #
    #     print("==================================")

    trees = model(sequence_feas,lengths=lengths)

    print('depth: ', len(trees))
    for level in trees:
        print(level[0].shape)
        print(level[1].sum(dim=-1))

