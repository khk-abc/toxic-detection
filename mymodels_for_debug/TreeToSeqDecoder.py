import torch
import torch.nn as nn
from .attentions import BertAttention


class TreetosqDecoder(nn.Module):
    def __init__(self,hidden_size=768,stack_num=3):
        super(TreetosqDecoder, self).__init__()

        self.stack_num=stack_num

        self.decoder = nn.ModuleList()
        for _ in range(stack_num):
            self.decoder.append(BertAttention(hidden_size=hidden_size,))

        self.decode_loss = nn.MSELoss()


    def _get_pad_attention_mask(self,paded_mask):

        pad_attention_mask = torch.matmul(paded_mask.unsqueeze(dim=-1),paded_mask.unsqueeze(dim=2))
        # print(pad_attention_mask)

        return  pad_attention_mask

    def forward(self,tree):
        tree_feas = [level[0][0] for level in tree[::-1]]
        pad_mask = [level[0][1] for level in tree[::-1]]

        device = tree_feas[0].device

        max_len_paded = max([i.shape[1] for i in tree_feas])

        for i in range(len(tree_feas)):
            if tree_feas[i].shape[1]<max_len_paded:
                bz,length,dim=tree_feas[i].shape


                tree_feas[i]=torch.cat([tree_feas[i],torch.zeros(bz,max_len_paded-length,dim).to(device)],dim=1)

                # for b in range(bz):
                #     print(pad_mask[i].sum(dim=-1)[b])
                #     print(pad_mask[i+1].sum(dim=-1)[b])
                #     pad_mask[i][b]=torch.cat([pad_mask[b][:pad_mask[i].sum(dim=-1)[b]],
                #                            torch.ones(pad_mask[i+1].sum(dim=-1)[b]-pad_mask[i].sum(dim=-1)[b]).to(device),
                #                            torch.zeros(max_len_paded-pad_mask[i+1].sum(dim=-1)[b]).to(device)],dim=1)

                pad_mask[i] = torch.cat([pad_mask[i + 1],
                                         torch.zeros(bz,max_len_paded - pad_mask[i + 1].shape[1]).to(device)],dim=1)

                # print(pad_mask[i])


        tree_feas = torch.stack(tree_feas,dim=1)  #  bz,depth,len,dim
        pad_mask = torch.stack(pad_mask,dim=1)

        pad_attention_mask = self._get_pad_attention_mask(paded_mask=pad_mask)

        input_feas = tree_feas[:,:-1,:,:].reshape(-1,max_len_paded,dim)
        input_attention_mask = pad_attention_mask[:,:-1,:,:].reshape(-1,max_len_paded,max_len_paded).unsqueeze(dim=1)
        # print(input_feas.shape)

        output_feas = tree_feas[:, 1:, :, :].reshape(-1,max_len_paded,dim)
        # print(output_feas.shape)

        # print(input_feas[0])
        # print(output_feas[0])
        # print(input_attention_mask)
        # print(pad_mask)



        # decoded_feas = self.decoder(input_feas,attention_mask=input_attention_mask)[0]
        decoded_feas = input_feas
        for i in range(self.stack_num):
            decoded_feas = self.decoder[i](decoded_feas, attention_mask=input_attention_mask)[0]



        restruct_loss = self.decode_loss(input=decoded_feas*pad_mask[:,:-1,:].reshape(-1,max_len_paded).unsqueeze(dim=-1),
                                         target=output_feas*pad_mask[:,:-1,:].reshape(-1,max_len_paded).unsqueeze(dim=-1))

        # print(input_feas)
        # print(output_feas)


        return restruct_loss



if __name__=='__main__':
    tree = [([torch.randn(2,5,8),torch.tensor([[1,1,1,1,0],[1,1,1,0,0]])],None),
            ([torch.randn(2,4,8),torch.tensor([[1,1,0,0],[1,1,1,0]])],None),
            ([torch.randn(2,1,8),torch.tensor([[1,],[1,]])],None)]

    model = TreetosqDecoder(hidden_size=8)

    restruct_loss=model(tree)

    print(restruct_loss)

    # print(tree_feas.shape)
    # print(tree_feas)

    # print(pad_attention_mask.shape)
    # print(pad_attention_mask)
