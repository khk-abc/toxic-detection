import torch
import torch.nn as nn

from .attentions import AttentionLayer, FullAttention, BertResidualOutput

from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence

class FeaTree(nn.Module):
    def __init__(self, device, stack_num=2, tree_dropout=0.):
        super(FeaTree, self).__init__()
        print('使用FeaTree v3!')

        self.stack_num = stack_num

        self.tree_dropout = tree_dropout
        if tree_dropout>0:
            self.dropout = nn.Dropout(tree_dropout)


        self.down_to_up_attention = nn.ModuleList()
        for _ in range(stack_num):
            self.down_to_up_attention.append(AttentionLayer(
                attention=FullAttention(attention_dropout=0.3, output_attention=True, ),
                d_model=768,
                n_heads=8,
            ))
        self.device = device

        self.residual_out = nn.ModuleList()
        for _ in range(stack_num):
            self.residual_out.append(BertResidualOutput(hidden_size=768,
                                                        layer_norm_eps=1e-10,
                                                        hidden_dropout_prob=0.3))



    def forward(self, init_tree, squence_feas, squence_pad_mask, role_query):


        new_tree = [([squence_feas, squence_pad_mask.to(self.device)], None), ]

        lower = [[squence_feas, squence_pad_mask.to(self.device)], ] + init_tree[:-1]

        for idx, (down, up) in enumerate(zip(lower, init_tree)):

            result = up[0]

            atten_scores = []
            for i in range(self.stack_num):
                if self.tree_dropout>0:
                    result = self.dropout(result)

                atten_out, atten_score = self.down_to_up_attention[i](queries=result, keys=down[0], values=down[0],
                                                                   query_mask=up[1],
                                                                   kv_mask=down[1],
                                                                   role_query=role_query,
                                                                   )

                result = self.residual_out[i](hidden_states=atten_out, input_tensor=result, )

                atten_scores.append(atten_score)


            new_up = ([result, up[1]], torch.stack(atten_scores, dim=1))

            new_tree.append(new_up)

        return new_tree


class TreeQuery(nn.Module):
    def __init__(self, device, stack_num=2,use_important=True,
                 delete_pad=False,tree_dropout=0.,learn_path=False):
        super(TreeQuery, self).__init__()
        print('TreeQuery v3!')

        self.top_to_bottom_attention = nn.ModuleList()
        for _ in range(stack_num):
            self.top_to_bottom_attention.append(AttentionLayer(
                attention=FullAttention(attention_dropout=0.3, output_attention=True, ),
                d_model=768,
                n_heads=8,
            ))

        self.stack_num = stack_num

        self.learn_path=learn_path
        if self.learn_path:
            self.path_learning = nn.LSTM(
                hidden_size=768,
                input_size=768,
                dropout=0.3,
                num_layers=1,
                bidirectional=False,
                batch_first=True
            )

        self.tree_dropout = tree_dropout
        if tree_dropout>0:
            self.dropout = nn.Dropout(tree_dropout)

        self.device = device

        self.residual_out = nn.ModuleList()
        for _ in range(stack_num):
            self.residual_out.append(BertResidualOutput(hidden_size=768,
                                                        layer_norm_eps=1e-10,
                                                        hidden_dropout_prob=0.3))


        self.use_important = use_important
        if use_important:
            print('使用重要性分数进行加权！')
            self.important_score = nn.Sequential(
                nn.Linear(768,768//2),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(768//2,1)
            )

        self.delete_pad = delete_pad

    def forward(self, query, tree, squence_pad_mask, role_query):
        """
        :param query:
        :param tree:[squence_fea,(current_level_fea,atten_scores),(...),...]
        squence_fea: bz,length,dim
        current_level_fea:bz,clust_num,dim
        atten_scores:bz,head_num,up_length,down_length
        :return:
        """
        if isinstance(tree[-1],tuple):
            query_histoty = []
            query_histoty.append((query + tree[-1][0][0]).squeeze(dim=1))  # 现将根节点添加

            query_node = ([query + tree[-1][0][0], tree[-1][0][1]], tree[-1][1])  # ([fea,pad_mask],atten_scores)
            assert tree[-1][0][0].shape[1]==1
        else:
            query_histoty = []
            query_histoty.append((query + tree[-1][0]).squeeze(dim=1))  # 现将根节点添加

            query_node = [query + tree[-1][0], tree[-1][1]]  # [fea,pad_mask]
            assert tree[-1][0].shape[1] == 1

        tree = tree[::-1]  # 从顶层到底层
        tree[0] = query_node

        attention_socres = []
        tree_level_result = []

        # decay_weights = torch.ones(len(tree)-1).to(query.device)
        decay_weights = torch.log(torch.arange(len(tree) - 1) / 4 + 1).to(query.device)
        if isinstance(tree[-1], tuple):
            for idx in range(len(tree)-1):
                result = tree[idx+1][0][0]  # node features

                atten_scores = []
                for i in range(self.stack_num):
                    if self.tree_dropout > 0:
                        result = self.dropout(result)
                    atten_out, atten_score = self.top_to_bottom_attention[i](
                        queries=result,
                        keys=tree[idx][0][0]+decay_weights[idx]*query,  # +query表示在每一层都加上查询进行“扰动”
                        values=tree[idx][0][0]+decay_weights[idx]*query,
                        query_mask=tree[idx+1][0][1],
                        kv_mask=tree[idx][0][1],
                        role_query=role_query,
                    )
                    result = self.residual_out[i](hidden_states=atten_out, input_tensor=result)
                    atten_scores.append(atten_score)


                tree[idx+1][0][0] = result

                atten_scores = torch.stack(atten_scores, dim=1)
                attention_socres.append(atten_scores)


                if self.use_important:
                    # 计算经过的节点的重要性
                    # query: [bz,1,dim]
                    # result: [bz,len,dim]
                    current_level_import_score = torch.sigmoid(self.important_score(query+result))
                    # current_level_import_score = torch.where(current_level_import_score >= 0.5,
                    #                                          current_level_import_score,
                    #                                          torch.zeros_like(current_level_import_score))
                    # current_level_import_score[current_level_import_score<0.5] = 0  # 重要性分数小于0.5赋值为0即不计入
                    if self.delete_pad:
                        tmp = result * current_level_import_score
                        tmp.masked_fill_(tree[idx + 1][0][1].unsqueeze(dim=-1) == 0, 0)
                        tree_level_result.append(tmp)
                        query_histoty.append((tmp).sum(dim=1))  # bz,dim
                    else:
                        tree_level_result.append(result * current_level_import_score)
                        query_histoty.append((result * current_level_import_score).sum(dim=1))  # bz,dim
                else:
                    if self.delete_pad:
                        result.masked_fill_(tree[idx + 1][0][1].unsqueeze(dim=-1) == 0, 0)
                        tree_level_result.append(result)
                    else:
                        tree_level_result.append(result)

            # print('result: ', result)
            # print('tree[-1][0][0]: ',tree[-1][0][0])
            assert torch.all(result==tree[-1][0][0])

            tree_level_result = torch.cat(tree_level_result,dim=1)  # 查询后所有节点，bz,lem,dim

            if self.use_important:
                query_histoty = torch.stack(query_histoty, dim=1)  # bz, level_num, dim
                if self.learn_path:
                    bz,lens,dims = query_histoty.shape
                    query_histoty = pack_padded_sequence(query_histoty,
                                                         lengths=[lens for _ in range(bz)],
                                                         batch_first=True)
                    query_histoty = self.path_learning(query_histoty)[0]
                    query_histoty,_ = pad_packed_sequence(query_histoty,batch_first=True)
                return (query_histoty, attention_socres, tree, tree_level_result)
            else:
                return (result, attention_socres, tree, tree_level_result)

            # 去除 PAD 部分
            # return (((result*tree[-1][0][1].unsqueeze(-1)).sum(dim=1)
            #          /tree[-1][0][1].sum(dim=-1,keepdim=True)).unsqueeze(dim=1), attention_socres, tree)

            # 累计多层
            # return (torch.stack([
            #     (item[0][0]*item[0][1].unsqueeze(-1)).sum(dim=1)/item[0][1].sum(dim=-1,keepdim=True)
            #     for item in tree],dim=1).mean(dim=1,keepdim=True), attention_socres, tree)

        else:
            for idx in range(len(tree)-1):
                result = tree[idx+1][0]  # node features

                atten_scores = []
                for i in range(self.stack_num):
                    atten_out, atten_score = self.top_to_bottom_attention[i](
                        queries=result,
                        keys=tree[idx][0]+decay_weights[idx]*query,
                        values=tree[idx][0]+decay_weights[idx]*query,
                        query_mask=tree[idx+1][1],
                        kv_mask=tree[idx][1],
                        role_query=role_query,
                    )
                    result = self.residual_out[i](hidden_states=atten_out, input_tensor=result)
                    atten_scores.append(atten_score)

                    if self.tree_dropout > 0:
                        result = self.dropout(result)

                tree[idx + 1][0] = result

                atten_scores = torch.stack(atten_scores, dim=1)
                attention_socres.append(atten_scores)


                if self.use_important:
                    # 计算经过的节点的重要性
                    # query: [bz,1,dim]
                    # result: [bz,len,dim]
                    current_level_import_score = torch.sigmoid(self.important_score(query + result))
                    # current_level_import_score = torch.where(current_level_import_score >= 0.5,
                    #                                          current_level_import_score,
                    #                                          torch.zeros_like(current_level_import_score))
                    # current_level_import_score[current_level_import_score < 0.5] = 0  # 重要性分数小于0.5赋值为0即不计入
                    if self.delete_pad:
                        tmp = result * current_level_import_score
                        tmp.masked_fill_(tree[idx + 1][1].unsqueeze(dim=-1) == 0, 0)
                        tree_level_result.append(tmp)
                        query_histoty.append((tmp).sum(dim=1))  # bz,dim
                    else:
                        tree_level_result.append(result * current_level_import_score)
                        query_histoty.append((result * current_level_import_score).sum(dim=1))  # bz,dim
                else:
                    if self.delete_pad:
                        result.masked_fill_(tree[idx + 1][1].unsqueeze(dim=-1) == 0, 0)
                        tree_level_result.append(result)
                    else:
                        tree_level_result.append(result)

            assert torch.all(result == tree[-1][0])

            tree_level_result = torch.cat(tree_level_result, dim=1)  # 查询后所有节点

            if self.use_important:
                query_histoty = torch.stack(query_histoty, dim=1) # bz, level_num, dim
                if self.learn_path:
                    bz,lens,dims = query_histoty.shape
                    query_histoty = pack_padded_sequence(query_histoty,
                                                         lengths=[lens for _ in range(bz)],
                                                         batch_first=True)
                    query_histoty = self.path_learning(query_histoty)[0]
                    query_histoty,_ = pad_packed_sequence(query_histoty,batch_first=True)
                return (query_histoty, attention_socres, tree,tree_level_result)
            else:
                return (result, attention_socres, tree,tree_level_result)


            # 去除 PAD 部分
            # return (((result * tree[-1][1].unsqueeze(-1)).sum(dim=1)
            #          / tree[-1][1].sum(dim=-1, keepdim=True)).unsqueeze(dim=1), attention_socres, tree)

            # 累计多层
            # return (torch.stack([
            #     (item[0] * item[1].unsqueeze(-1)).sum(dim=1) / item[1].sum(dim=-1, keepdim=True)
            #     for item in tree], dim=1).mean(dim=1, keepdim=True), attention_socres, tree)

