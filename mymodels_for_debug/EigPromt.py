import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.matrices import Matrix, GramSchmidt
from .attentions import *
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

class EigPrompter(nn.Module):
    def __init__(self, feature_size, config, myargs=None, query_style='attention', eig_num=10, use_orthogo=False):
        super(EigPrompter, self).__init__()

        self.eig_num = eig_num
        self.myargs = myargs

        if myargs.enhance_init_style == 'definition':
            print('使用定义增强！')
            if myargs.task_prompt_output == 'embeddings_emb':
                if myargs.emb_model == 'bert':
                    output = torch.load(
                        'DataLoaders/definitions_from_embeddings_v1.ckpt')
                elif myargs.emb_model == 'roberta':
                    output = torch.load(
                        'DataLoaders/definitions_from_embeddings_v1_roberta.ckpt')
                elif myargs.emb_model == 'hateBERT':
                    output = torch.load(
                        'DataLoaders/definitions_v1_from_embeddings_hatebert.ckpt')


                self.eig_num = output.shape[0]
                self.eigprompt = nn.Embedding.from_pretrained(output.mean(dim=1).detach(),
                                                              freeze=False)
                self.eigprompt.weight.requires_grad = True
            elif myargs.task_prompt_output == 'bert-with-eval':
                if myargs.emb_model == 'bert':
                    output = torch.load(
                        'DataLoaders/definitions_v1.ckpt')
                elif myargs.emb_model == 'roberta':
                    output = torch.load(
                        'DataLoaders/definitions_v1_roberta.ckpt')

                elif myargs.emb_model == 'hateBERT':
                    output = torch.load(
                        'DataLoaders/definitions_v1_hatebert.ckpt')

                self.eig_num = output['last_hidden_state'].shape[0]
                self.eigprompt = nn.Embedding.from_pretrained(output['last_hidden_state'].mean(dim=1).detach(),
                                                              freeze=False)
                self.eigprompt.weight.requires_grad = True

        elif myargs.enhance_init_style == 'orthogo':
            print('使用正交！')
            init_weights = torch.randn(eig_num, feature_size)
            ort_tensor = self.init_eigprompt(init_weights)
            self.eigprompt = nn.Embedding.from_pretrained(ort_tensor, freeze=False)
            self.eigprompt.weight.requires_grad = True
        else:
            self.eigprompt = nn.Embedding(eig_num, feature_size)  # weight: shape[eig_num,feature_size]

        if query_style == 'attention':
            self.queryer = BertSelfAttention(config=config)
            # self.queryer = BertAttention(config=config)

    def init_eigprompt(self, weight):
        m, n = weight.size()
        weight_np = weight.detach().numpy()
        matrix = [Matrix(col) for col in weight_np]
        gram = GramSchmidt(matrix)
        # ort_list = np.array(gram)
        ort_list = []
        for i in range(m):
            vector = []
            for j in range(n):
                vector.append(float(gram[i][j]))
            ort_list.append(vector)
        ort_list = np.mat(ort_list)
        ort_np = torch.from_numpy(ort_list)
        ort_tensor = F.normalize(ort_np, dim=1)

        return ort_tensor

    def forward(self, query_features):
        key_features = self.eigprompt(torch.arange(0, self.eig_num).to(query_features.device))
        # print(query_features.shape) # task_num,hidden_size
        # print(key_features.shape)  # eig_num,hidden_size
        query_features = query_features
        key_features = key_features.unsqueeze(0)
        query_out = self.queryer(hidden_states=query_features,
                                 encoder_hidden_states=key_features.type_as(query_features))

        return query_out
