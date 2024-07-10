import torch.nn.functional as F
# from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig
from sympy.matrices import Matrix, GramSchmidt
from transformers import AutoModel

from multiprocessing import Pool

from .SupervisedContrastLearning import SupConLoss
from .TreeToSeqDecoder import TreetosqDecoder
from .attentions import *
from .get_tree import TreeGetter
from .utils import get_tree
from .EigPromt import EigPrompter
from .FFNs import TwoLayerFFNNLayer
from .OptimalTransport_debug import OptimalTransport

import torch.autograd as autograd


# from .featree import FeaTree, TreeQuery
# from .featree_v2 import FeaTree, TreeQuery


def normalize(features):
    mean = features.detach().mean(dim=-1,keepdim=True)
    std = features.detach().std(dim=-1,keepdim=True)

    return torch.div((features-mean),std)


def choose_featree_version(version='v2'):
    if version == 'v0':
        from .featree import FeaTree, TreeQuery
        return FeaTree, TreeQuery
    elif version == 'v1':
        from .featree_v1 import FeaTree, TreeQuery
        return FeaTree, TreeQuery
    elif version == 'v2':
        from .featree_v2 import FeaTree, TreeQuery
        return FeaTree, TreeQuery
    elif version == 'v3':
        from .featree_v3 import FeaTree, TreeQuery
        return FeaTree, TreeQuery
    else:
        raise ValueError('目前实现v0-v3，请从其中选择！')


class TotalModel(nn.Module):
    def __init__(self, config, myargs, use_toxic_enhance=False, use_tree=True):
        super(TotalModel, self).__init__()

        self.device = config.device

        self.TASK_NUM = config.TASK_NUM

        self.embed_model = AutoModel.from_pretrained(config.model_name)

        # self.freeze(module=self.embed_model)
        if myargs.use_dropout:
            # TODO: 2024年1月22日加上dropout
            self.dropout = nn.Dropout(config.dropout)

        self.use_toxic_enhance = myargs.use_toxic_enhance

        # TODO: 加上任务embedding
        if myargs.task_query_embedding:
            self.task_query_embedding = nn.Embedding(self.TASK_NUM, 768)

        self.config = config
        self.myargs = myargs

        FeaTree, TreeQuery = choose_featree_version(version=myargs.treequery_version)

        if self.myargs.use_tree and self.myargs.construct_tree_style == 'conv':
            print('基于卷积使用树！')
            self.treegetter = TreeGetter()
            self.featree = FeaTree(
                device=config.device,
                stack_num=self.myargs.stack_num,
                tree_dropout=self.myargs.tree_dropout
            )
            self.treequery = TreeQuery(device=config.device,
                                       use_important=self.myargs.use_important_score,
                                       delete_pad=self.myargs.delete_pad,
                                       tree_dropout=self.myargs.tree_dropout,
                                       learn_path=self.myargs.learn_path,
                                       stack_num=self.myargs.stack_num,
                                       )
            if self.myargs.split_gpt_sample:
                if self.myargs.gptall:
                    self.gpt_treegetter = TreeGetter()
                    self.gpt_featree = FeaTree(device=config.device,
                                               stack_num=self.myargs.stack_num,
                                               tree_dropout=self.myargs.tree_dropout)
                self.gpt_treequery = TreeQuery(device=config.device,
                                               use_important=self.myargs.use_important_score,
                                               delete_pad=self.myargs.delete_pad,
                                               tree_dropout=self.myargs.tree_dropout,
                                               learn_path=self.myargs.learn_path,
                                               stack_num=self.myargs.stack_num,
                                               )
        elif self.myargs.use_tree and self.myargs.construct_tree_style == 'kmeans':
            print("基于聚类使用树！")

        self.tree_decoder = None
        if self.myargs.use_decoder:
            print('使用解码器！')
            self.tree_decoder = TreetosqDecoder(hidden_size=768)

        self.use_same_level_contrast = self.myargs.use_same_level_contrast
        if self.myargs.use_same_level_contrast:
            self.contraster = SupConLoss(config=config, temperature=0.5, scale_by_temperature=True)
            print('使用同级对比！')

        self.classifier = TwoLayerFFNNLayer(config, myargs=myargs)

        if not myargs.not_extra_enhance:
            if self.myargs.use_extra_share:
                print('使用任务问句对任务token进行扩展！')
            if self.myargs.enhace_level == 'sent':
                print('使用共享特征池对任务token进行增强！')
                self.eigprompter = EigPrompter(feature_size=config.hidden_size, config=config, myargs=myargs)

        if self.myargs.use_toxic_enhance:
            print('使用毒性增强！')
            self.toxic_embedding = nn.Embedding(6, config.hidden_size)

        if self.myargs.notask_in_tree:
            # notask_in_tree
            print('任务token不参与构建树！')

        if self.myargs.splitbatch:
            print('一个样本一个样本处理！')

        if myargs.use_soft_alpha > 0:
            print('使用gpt软标签！')

        if myargs.use_gpt_tree:
            print("使用gpt构造树！")

        if myargs.use_OT:
            #TODO
            self.ot_source_linears = nn.ModuleList()
            self.ot_target_linears = nn.ModuleList()
            for _ in range(self.TASK_NUM):
                self.ot_source_linears.append(nn.Sequential(
                    nn.Linear(768,768//2),
                    nn.Tanh(),
                    nn.Dropout(0.5),
                    nn.Linear(768//2,1),
                    nn.Sigmoid()
                ))
                self.ot_target_linears.append(nn.Sequential(
                    nn.Linear(768,768//2),
                    nn.Tanh(),
                    nn.Dropout(0.5),
                    nn.Linear(768//2,1),
                    nn.Sigmoid()
                ))

            assert myargs.use_gpt_tree
            print('使用最优传输！')
            self.ot = OptimalTransport(
                train_cost_matrix=myargs.train_cost_matrix,
                train_cost_matrix_v2=myargs.train_cost_matrix_v2,
            )

            if self.myargs.trainable_epislon:
                self.epsilon = nn.Parameter(torch.tensor([1e-2 for _ in range(self.TASK_NUM)]), requires_grad=True)
            if self.myargs.trainable_Ot_alpha:
                if myargs.use_two_side:
                    self.ot_alpha_toxic = nn.Parameter(torch.tensor([0.2 for _ in range(self.TASK_NUM)]),
                                                       requires_grad=True)
                    self.ot_alpha_non_toxic = nn.Parameter(torch.tensor([0.2 for _ in range(self.TASK_NUM)]),
                                                           requires_grad=True)
                else:
                    self.ot_alpha = nn.Parameter(torch.tensor([0.2 for _ in range(self.TASK_NUM)]), requires_grad=True)
            elif self.myargs.learn_ot_alpha:
                if myargs.use_two_side:
                    self.ot_alpha_toxic = nn.ModuleList()
                    self.ot_alpha_non_toxic = nn.ModuleList()
                    for i in range(self.TASK_NUM):
                        self.ot_alpha_toxic.append(nn.Sequential(
                            nn.Linear(768, 1),
                            nn.Sigmoid()
                        ))
                        self.ot_alpha_non_toxic.append(nn.Sequential(
                            nn.Linear(768, 1),
                            nn.Sigmoid()
                        ))
                else:
                    self.ot_alpha = nn.ModuleList()
                    for i in range(self.TASK_NUM):
                        self.ot_alpha.append(nn.Sequential(
                            nn.Linear(768, 1),
                            nn.Sigmoid()
                        ))

        if myargs.data_version == 'with_range':
            print('使用带范围提示的gpt背景！')
        elif myargs.data_version == 'two_side':
            print("使用双边gpt背景！")
        elif myargs.data_version == 'all_tasks':
            print('使用所有任务gpt背景！')
        elif myargs.data_version == 'all_tasks_modified':
            print('使用修正后的所有任务gpt背景！')

    def freeze(self, module):
        for n, p in module.named_parameters():
            p.requires_grad = False

    def _get_inittree_use_cluster(self, seq_fea_init, attention_mask):
        # 获取特征树
        with torch.no_grad():
            seq_fea_init = seq_fea_init.detach().cpu()
            lengths = torch.sum(attention_mask, dim=-1)

            # means = seq_fea_init.mean(dim=-1,keepdim=True)
            # stds = seq_fea_init.std(dim=-1,keepdim=True)
            # seq_fea_init = torch.div((seq_fea_init-means),stds)*input_batch['attention_mask'].unsqueeze(-1).cpu()

            init_tree = get_tree(sequence_feas=seq_fea_init, lengths=lengths.numpy().tolist(), k=4)
            init_tree = [[level[0].to(self.device), level[1].to(self.device)] for level in init_tree]

        return init_tree

    def get_gpt_context_tree(self, input_batch):
        assert self.myargs.use_gpt_tree

        batchsize = input_batch['gpt_context_input_ids'].shape[0]

        final_output = self.embed_model(input_ids=input_batch['gpt_context_input_ids'].to(self.device),
                                        token_type_ids=input_batch['gpt_context_token_type_ids'].to(self.device),
                                        attention_mask=input_batch['gpt_context_attention_mask'].to(self.device))

        if self.myargs.construct_tree_style == 'kmeans':
            init_tree = self._get_inittree_use_cluster(final_output['last_hidden_state'],
                                                       input_batch['gpt_context_attention_mask'])
            sequence_feas = final_output['last_hidden_state']
            squence_pad_mask = input_batch['gpt_context_attention_mask']
        elif self.myargs.construct_tree_style == 'conv':
            lengths = torch.sum(input_batch['gpt_context_attention_mask'], dim=-1)
            if self.myargs.gptall:
                init_tree = self.gpt_treegetter(sequence_feas=final_output['last_hidden_state'], lengths=lengths)
            else:
                init_tree = self.treegetter(sequence_feas=final_output['last_hidden_state'], lengths=lengths)
            sequence_feas = final_output['last_hidden_state']
            squence_pad_mask = input_batch['gpt_context_attention_mask']

        if self.myargs.gptall:
            gpt_context_tree = self.gpt_featree(
                init_tree=init_tree,
                squence_feas=sequence_feas,
                squence_pad_mask=squence_pad_mask,
                role_query=False,
            )
        else:
            gpt_context_tree = self.featree(
                init_tree=init_tree,
                squence_feas=sequence_feas,
                squence_pad_mask=squence_pad_mask,
                role_query=False,
            )
        return gpt_context_tree, final_output

    def get_toxic_gpt_context_tree(self, input_batch):
        assert self.myargs.use_gpt_tree

        batchsize = input_batch['toxic_gpt_context_input_ids'].shape[0]

        final_output = self.embed_model(input_ids=input_batch['toxic_gpt_context_input_ids'].to(self.device),
                                        token_type_ids=input_batch['toxic_gpt_context_token_type_ids'].to(self.device),
                                        attention_mask=input_batch['toxic_gpt_context_attention_mask'].to(self.device))

        if self.myargs.construct_tree_style == 'kmeans':
            init_tree = self._get_inittree_use_cluster(final_output['last_hidden_state'],
                                                       input_batch['toxic_gpt_context_attention_mask'])
            sequence_feas = final_output['last_hidden_state']
            squence_pad_mask = input_batch['toxic_gpt_context_attention_mask']
        elif self.myargs.construct_tree_style == 'conv':
            lengths = torch.sum(input_batch['toxic_gpt_context_attention_mask'], dim=-1)
            if self.myargs.gptall:
                init_tree = self.gpt_treegetter(sequence_feas=final_output['last_hidden_state'], lengths=lengths)
            else:
                init_tree = self.treegetter(sequence_feas=final_output['last_hidden_state'], lengths=lengths)
            sequence_feas = final_output['last_hidden_state']
            squence_pad_mask = input_batch['toxic_gpt_context_attention_mask']

        if self.myargs.gptall:
            toxic_gpt_context_tree = self.gpt_featree(
                init_tree=init_tree,
                squence_feas=sequence_feas,
                squence_pad_mask=squence_pad_mask,
                role_query=False,
            )
        else:
            toxic_gpt_context_tree = self.featree(
                init_tree=init_tree,
                squence_feas=sequence_feas,
                squence_pad_mask=squence_pad_mask,
                role_query=False,
            )
        return toxic_gpt_context_tree, final_output

    def get_non_toxic_gpt_context_tree(self, input_batch):
        assert self.myargs.use_gpt_tree

        batchsize = input_batch['non_toxic_gpt_context_input_ids'].shape[0]

        final_output = self.embed_model(input_ids=input_batch['non_toxic_gpt_context_input_ids'].to(self.device),
                                        token_type_ids=input_batch['non_toxic_gpt_context_token_type_ids'].to(
                                            self.device),
                                        attention_mask=input_batch['non_toxic_gpt_context_attention_mask'].to(
                                            self.device))

        if self.myargs.construct_tree_style == 'kmeans':
            init_tree = self._get_inittree_use_cluster(final_output['last_hidden_state'],
                                                       input_batch['non_toxic_gpt_context_attention_mask'])
            sequence_feas = final_output['last_hidden_state']
            squence_pad_mask = input_batch['non_toxic_gpt_context_attention_mask']
        elif self.myargs.construct_tree_style == 'conv':
            lengths = torch.sum(input_batch['non_toxic_gpt_context_attention_mask'], dim=-1)
            if self.myargs.gptall:
                init_tree = self.gpt_treegetter(sequence_feas=final_output['last_hidden_state'], lengths=lengths)
            else:
                init_tree = self.treegetter(sequence_feas=final_output['last_hidden_state'], lengths=lengths)
            sequence_feas = final_output['last_hidden_state']
            squence_pad_mask = input_batch['non_toxic_gpt_context_attention_mask']

        if self.myargs.gptall:
            non_toxic_gpt_context_tree = self.gpt_featree(
                init_tree=init_tree,
                squence_feas=sequence_feas,
                squence_pad_mask=squence_pad_mask,
                role_query=False,
            )
        else:
            non_toxic_gpt_context_tree = self.featree(
                init_tree=init_tree,
                squence_feas=sequence_feas,
                squence_pad_mask=squence_pad_mask,
                role_query=False,
            )
        return non_toxic_gpt_context_tree, final_output
        # return non_toxic_gpt_context_tree

    def get_sample_content_tree(self, input_batch, final_output):
        batchsize = input_batch['input_ids'].shape[0]

        if self.myargs.notask_in_tree and self.myargs.construct_tree_style == 'conv':
            # notask_in_tree
            sequence_feas = torch.clone(final_output['last_hidden_state']).to(self.device)
            sequence_feas[torch.arange(0, batchsize).unsqueeze(-1), input_batch['index']] = \
                torch.zeros(batchsize, self.TASK_NUM, 768).to(self.device)
            squence_pad_mask = torch.clone(input_batch['attention_mask']).to(self.device)
            squence_pad_mask[torch.arange(0, batchsize).unsqueeze(-1), input_batch['index']] = \
                torch.zeros(batchsize, self.TASK_NUM).to(self.device).long()
            lengths = torch.sum(squence_pad_mask, dim=-1)
            init_tree = self.treegetter(sequence_feas=sequence_feas, lengths=lengths)
        elif self.myargs.construct_tree_style == 'kmeans':
            init_tree = self._get_inittree_use_cluster(final_output['last_hidden_state'],
                                                       input_batch['attention_mask'])
            sequence_feas = final_output['last_hidden_state']
            squence_pad_mask = input_batch['attention_mask']
        elif (not self.myargs.notask_in_tree) and self.myargs.construct_tree_style == 'conv':
            # last_hidden_states
            lengths = torch.sum(input_batch['attention_mask'], dim=-1)
            init_tree = self.treegetter(sequence_feas=final_output['last_hidden_state'], lengths=lengths)
            sequence_feas = final_output['last_hidden_state']
            squence_pad_mask = input_batch['attention_mask']

        sample_content_tree = self.featree(
            init_tree=init_tree,
            squence_feas=sequence_feas,
            squence_pad_mask=squence_pad_mask,
            role_query=False,
        )

        return sample_content_tree

    def get_task_prompt_out(self, task_prompt):
        if self.myargs.task_prompt_output == 'embeddings_emb':
            # print('使用embedding层表示！')
            with torch.no_grad():
                self.embed_model.embeddings.eval()
                task_prompt_output = self.embed_model.embeddings(task_prompt['input_ids'].to(self.device)).detach()
                self.embed_model.embeddings.train()
            task_prompt_output = task_prompt_output * task_prompt['attention_mask'].to(self.device).unsqueeze(-1)
            # task_prompt_output = task_prompt_output.sum(dim=1)
            # task_prompt_output = task_prompt_output/task_prompt['attention_mask'].unsqueeze(-1).sum(dim=1)
        elif self.myargs.task_prompt_output == 'bert-with-eval':
            # print('使用bert表示！')
            with torch.no_grad():
                self.embed_model.eval()
                task_prompt_output = self.embed_model(input_ids=task_prompt['input_ids'].to(self.device),
                                                      token_type_ids=task_prompt['token_type_ids'].to(self.device),
                                                      attention_mask=task_prompt['attention_mask'].to(self.device))[
                    'last_hidden_state'].detach()
                task_prompt_output = task_prompt_output * task_prompt['attention_mask'].to(self.device).unsqueeze(
                    -1)
                self.embed_model.train()
        else:
            # print('使用bert表示！')
            task_prompt_output = self.embed_model(input_ids=task_prompt['input_ids'].to(self.device),
                                                  token_type_ids=task_prompt['token_type_ids'].to(self.device),
                                                  attention_mask=task_prompt['attention_mask'].to(self.device))[
                'last_hidden_state']
            task_prompt_output = task_prompt_output * task_prompt['attention_mask'].to(self.device).unsqueeze(-1)
        # print(task_prompt_output)

        task_prompt_output = task_prompt_output.mean(dim=1).unsqueeze(0)  # 1,self.TASK_NUM,dim

        return task_prompt_output

    def get_content_out(self, input_batch, task_prompt, analysis=False):
        batchsize = input_batch['input_ids'].shape[0]

        # 获取文本的表示
        inputs_embeds = self.embed_model.embeddings.word_embeddings(input_batch['input_ids'].to(self.device))
        # with torch.no_grad():
        #     inputs_embeds = self.embed_model.embeddings.word_embeddings(input_batch['input_ids'].to(self.device))
        # 是否采用毒性增强，默认不采用
        if self.use_toxic_enhance:
            toxic_embedding = self.toxic_embedding(toxic_ids=input_batch["toxic_ids"].to(self.device))
            inputs_embeds = toxic_embedding + inputs_embeds

        if not self.myargs.not_extra_enhance:
            if self.myargs.use_extra_share:
                # 使用任务特定问句对任务对应token进行增强
                # 获取任务相关prompt的表示
                task_prompt_output = self.get_task_prompt_out(task_prompt=task_prompt)
            else:
                # 不使用任务特定问句对任务对应token进行增强
                # bz,self.TASK_NUM,dim
                task_prompt_output = inputs_embeds[torch.arange(0, batchsize).unsqueeze(-1), input_batch['index']]

            # 通过任务相关prompt与任务定义之间的查询得到结合共享特征的任务prompt特征和定义特征
            # task_prompt_output shape self.TASK_NUM,dim
            if self.myargs.enhace_level == 'sent':
                # print('使用sent query！')
                query_out = self.eigprompter(query_features=task_prompt_output)[0]

            else:
                query_out = task_prompt_output

            # # 将文本中对应位置的token用增强后的进行更新
            # # pre update
            # inputs_embeds[torch.arange(0, batchsize).unsqueeze(-1), input_batch['index']] = query_out
            inputs_embeds[torch.arange(0, batchsize).unsqueeze(-1), input_batch['index']] = \
                (inputs_embeds[torch.arange(0, batchsize).unsqueeze(-1), input_batch['index']] + query_out) / 2

        # 获取最终表示
        final_output = self.embed_model(inputs_embeds=inputs_embeds,
                                        token_type_ids=input_batch['token_type_ids'].to(self.device),
                                        attention_mask=input_batch['attention_mask'].to(self.device))

        restruct_loss = None
        contrat_loss = None
        toxic_con_loss, toxic_type_con_loss = None, None
        cost_distances = None

        if self.myargs.use_tree:
            sample_content_tree = self.get_sample_content_tree(input_batch=input_batch, final_output=final_output)

            if self.myargs.use_gpt_tree:
                if self.myargs.use_two_side:
                    toxic_gpt_context_tree, toxic_gpt_final_output = self.get_toxic_gpt_context_tree(
                        input_batch=input_batch)
                    non_toxic_gpt_context_tree, non_toxic_gpt_final_output = self.get_non_toxic_gpt_context_tree(
                        input_batch=input_batch)
                else:
                    gpt_context_tree, gpt_final_output = self.get_gpt_context_tree(input_batch=input_batch)

            # 利用task对应token特征从树中查询结果
            # bz，self.TASK_NUM，768
            query_tree_fea = final_output['last_hidden_state'][
                torch.arange(0, batchsize).unsqueeze(-1), input_batch['index']]
            if self.myargs.task_query_embedding:
                query_tree_fea = query_tree_fea + \
                                 self.task_query_embedding(
                                     torch.arange(self.TASK_NUM).repeat(batchsize, 1).to(self.device)
                                 )

            # 将文本中对应位置的token用增强后的进行更新，即查询特征包括本身语义特征、问题特征和任务定义特征
            # # post update
            # query_tree_fea = (query_tree_fea + query_out) / 2

            # 针对样例树进行查询
            sample_results = []
            sample_query_res_attention_scores = []
            sample_query_res_feas = []
            sample_tree_after_query = []

            # 针对gpt背景树进行查询
            if self.myargs.use_two_side:
                toxic_gpt_context_results = []
                toxic_gpt_context_query_res_attention_scores = []
                toxic_gpt_context_query_res_feas = []
                toxic_gpt_context_tree_after_query = []

                non_toxic_gpt_context_results = []
                non_toxic_gpt_context_query_res_attention_scores = []
                non_toxic_gpt_context_query_res_feas = []
                non_toxic_gpt_context_tree_after_query = []
            else:
                gpt_context_results = []
                gpt_context_query_res_attention_scores = []
                gpt_context_query_res_feas = []
                gpt_context_tree_after_query = []
            # cost_distances = 0
            for current_task in range(self.TASK_NUM):
                # 样例查询
                if self.myargs.not_clone_tree:
                    sample_content_tree_copy = sample_content_tree
                else:
                    sample_content_tree_copy = [([torch.clone(item[0][0]), item[0][1]], item[1]) for item in
                                                sample_content_tree]
                sample_full_result = self.treequery(
                    query=query_tree_fea[:, current_task, :].unsqueeze(1),
                    tree=sample_content_tree_copy, squence_pad_mask=input_batch['attention_mask'],
                    role_query=True,
                )

                sample_query_res_feas.append(sample_full_result[0])
                sample_query_res_attention_scores.append(sample_full_result[1])
                sample_tree_after_query.append(sample_full_result[2])

                sample_result = sample_full_result[0]
                # sample_results.append(sample_result.mean(dim=1) + query_tree_fea[:, current_task, :])

                if self.myargs.use_gpt_tree:
                    # gpt查询
                    if self.myargs.not_clone_tree:
                        gpt_context_content_tree_copy = gpt_context_tree
                    else:
                        gpt_context_content_tree_copy = [([torch.clone(item[0][0]), item[0][1]], item[1]) for item
                                                         in gpt_context_tree]

                    if self.myargs.split_gpt_sample:
                        gpt_context_full_result = self.gpt_treequery(
                            query=query_tree_fea[:, current_task, :].unsqueeze(1),
                            tree=gpt_context_content_tree_copy,
                            # squence_pad_mask=input_batch['attention_mask'],
                            squence_pad_mask=input_batch['gpt_context_attention_mask'],
                            # TODO 修改bug：attention_mask——>gpt_context_attention_mask
                            role_query=True,
                        )
                    else:
                        gpt_context_full_result = self.treequery(
                            query=query_tree_fea[:, current_task, :].unsqueeze(1),
                            tree=gpt_context_content_tree_copy,
                            # squence_pad_mask=input_batch['attention_mask'],
                            squence_pad_mask=input_batch['gpt_context_attention_mask'],
                            # TODO 修改bug：attention_mask——>gpt_context_attention_mask
                            role_query=True,
                        )
                    # gpt_context_query_res_feas.append(gpt_context_full_result[0])
                    # gpt_context_query_res_attention_scores.append(gpt_context_full_result[1])
                    # gpt_context_tree_after_query.append(gpt_context_full_result[2])
                    gpt_context_result = gpt_context_full_result[0]

                    if self.myargs.use_OT:
                        bz, target_len, dims = sample_result.shape
                        bz, source_len, dims = gpt_context_result.shape
                        # TODO epsilon --> trainable
                        if self.myargs.cat_pooler:
                            gpt_context_result = torch.cat([gpt_final_output['pooler_output'].unsqueeze(dim=1),
                                                            gpt_context_result], dim=1)
                            sample_result = torch.cat(
                                [final_output['pooler_output'].unsqueeze(dim=1),
                                 sample_result], dim=1)

                        # TODO : DEBUG 2024,01,26 改变传输尺寸
                        gpt_source = self.ot_source_linears[current_task](
                            gpt_context_result).squeeze(-1)
                        sample_target = self.ot_target_linears[current_task](
                            sample_result).squeeze(-1)

                        # gpt_source = self.ot_source_linears[current_task](
                        #     torch.stack([gpt_context_result.mean(dim=-1),gpt_context_result.std(dim=-1)],dim=-1)
                        # ).squeeze(-1)
                        # sample_target = self.ot_target_linears[current_task](
                        #     torch.stack([sample_result.mean(dim=-1),sample_result.std(dim=-1)],dim=-1)
                        # ).squeeze(-1)


                        # with torch.no_grad():
                        cost_matrix = self.ot._comput_cost_matrix_batch(source_features=gpt_context_result,
                                                                        target_features=sample_result,
                                                                        # task_fea=task_fea,
                                                                        task_id=current_task,
                                                                        metric=self.myargs.metric,
                                                                        use_kernel_cost=self.myargs.use_kernel_cost,
                                                                        )
                        cost_distance, trasport_matrix, cost_matrix, U, V = self.ot(
                            # source_features=gpt_context_result,
                            # target_features=sample_result,
                            source_features=gpt_source,
                            target_features=sample_target,
                            cost_matrix=cost_matrix,
                            # task_fea=query_tree_fea[:, current_task,
                            #          :],
                            task_id=current_task,
                            epsilon=1e-2 if not self.myargs.trainable_epislon else self.epsilon[current_task],
                            thresh=self.myargs.thresh,
                            max_iter=self.myargs.max_iter,
                            use_OT_kernel=self.myargs.use_OT_kernel,
                            speedup=self.myargs.speedup,
                            detach=self.myargs.detach,
                            metric=self.myargs.metric,
                            use_kernel_cost=self.myargs.use_kernel_cost
                        )

                        cost_distance = cost_distance.mean()
                        # cost_distances += cost_distance / self.TASK_NUM
                        # TODO: use_OT_alpha --> trainable parameters
                        if self.myargs.trainable_Ot_alpha:
                            if not self.myargs.normal_ot_scores:
                                sample_result = (sample_result
                                                 + self.ot_alpha[current_task] *
                                                 torch.einsum('nst,nsd->ntd',
                                                              trasport_matrix,
                                                              gpt_context_result)) / (
                                                            1 + self.ot_alpha[current_task])
                            else:
                                sample_result = (sample_result +
                                                 self.ot_alpha[current_task] *
                                                 torch.einsum('nst,nsd->ntd',
                                                              trasport_matrix,
                                                              gpt_context_result) /
                                                 trasport_matrix.sum(dim=1)) / (
                                                            1 + self.ot_alpha[current_task])
                        elif self.myargs.learn_ot_alpha:
                            task_ot_alpha = self.ot_alpha[current_task](sample_result.reshape(bz, dims, -1).permute(0, 2, 1).mean(dim=1).mean(dim=1) +
                                                             gpt_context_result.reshape(bz, dims, -1).permute(0, 2, 1).mean(dim=1).mean(dim=1) +
                                                             query_tree_fea[:, current_task, :])
                            if not self.myargs.normal_ot_scores:
                                sample_result = (sample_result +
                                                 task_ot_alpha.unsqueeze(dim=1) *
                                                 torch.einsum('nst,nsd->ntd',
                                                              trasport_matrix,
                                                              gpt_context_result)) / (
                                                        1 + task_ot_alpha.unsqueeze(dim=1))
                            else:
                                sample_result = (sample_result +
                                                 task_ot_alpha.unsqueeze(dim=1) *
                                                 torch.einsum('nst,nsd->ntd',
                                                              trasport_matrix,
                                                              gpt_context_result) /
                                                 trasport_matrix.sum(dim=1)) / \
                                                (1 + task_ot_alpha.unsqueeze(dim=1))
                        else:
                            if not self.myargs.normal_ot_scores:
                                # print(trasport_matrix.shape)
                                sample_result = (1 - self.myargs.use_OT_alpha) * sample_result + \
                                                self.myargs.use_OT_alpha * torch.einsum('nst,nsd->ntd',
                                                                                        trasport_matrix,
                                                                                        gpt_context_result)
                            else:
                                sample_result = (1 - self.myargs.use_OT_alpha) * sample_result + \
                                                self.myargs.use_OT_alpha * \
                                                torch.einsum('nst,nsd->ntd',
                                                             trasport_matrix,
                                                             gpt_context_result) / trasport_matrix.sum(dim=1)

                        # 尺寸转换回去
                        # sample_result = sample_result.reshape(bz,dims,-1).permute(0,2,1)
                        # gpt_context_result = gpt_context_result.reshape(bz,dims,-1).permute(0,2,1)
                    else:
                        # 不使用最优传输，直接将gpt结果加到样例中
                        sample_result = sample_result.mean(dim=1, keepdim=True) + gpt_context_result.mean(dim=1,
                                                                                                          keepdim=True)

                    if self.myargs.gpt_tree_supervised:
                        gpt_context_results.append(gpt_context_result.mean(dim=1) + query_tree_fea[:, current_task, :])

                sample_results.append(sample_result.mean(dim=1) + query_tree_fea[:, current_task, :])

            if self.myargs.use_dropout:
                for idx in range(len(sample_results)):
                    if self.myargs.add_pooler:
                        sample_results[idx] = sample_results[idx] + final_output['pooler_output']
                    sample_results[idx] = self.dropout(sample_results[idx])
            elif self.myargs.add_pooler:
                for idx in range(len(sample_results)):
                    sample_results[idx] = sample_results[idx] + final_output['pooler_output']

            results = torch.stack(sample_results, dim=1)
            if self.myargs.gpt_tree_supervised:
                if self.myargs.use_two_side:

                    if self.myargs.use_dropout:
                        for idx in range(len(toxic_gpt_context_results)):
                            if self.myargs.add_pooler:
                                toxic_gpt_context_results[idx] = toxic_gpt_context_results[idx] + \
                                                                 toxic_gpt_final_output['pooler_output']
                                non_toxic_gpt_context_results[idx] = non_toxic_gpt_context_results[idx] + \
                                                                     non_toxic_gpt_final_output['pooler_output']
                            toxic_gpt_context_results[idx] = self.dropout(toxic_gpt_context_results[idx])
                            non_toxic_gpt_context_results[idx] = self.dropout(non_toxic_gpt_context_results[idx])
                    elif self.myargs.add_pooler:
                        for idx in range(len(toxic_gpt_context_results)):
                            toxic_gpt_context_results[idx] = toxic_gpt_context_results[idx] + toxic_gpt_final_output[
                                'pooler_output']
                            non_toxic_gpt_context_results[idx] = non_toxic_gpt_context_results[idx] + \
                                                                 non_toxic_gpt_final_output['pooler_output']

                    toxic_gpt_results = torch.stack(toxic_gpt_context_results, dim=1)
                    non_toxic_gpt_results = torch.stack(non_toxic_gpt_context_results, dim=1)
                else:

                    if self.myargs.use_dropout:
                        for idx in range(len(gpt_context_results)):
                            if self.myargs.add_pooler:
                                gpt_context_results[idx] = gpt_context_results[idx] + gpt_final_output['pooler_output']
                            gpt_context_results[idx] = self.dropout(gpt_context_results[idx])
                    elif self.myargs.add_pooler:
                        for idx in range(len(gpt_context_results)):
                            gpt_context_results[idx] = gpt_context_results[idx] + gpt_final_output['pooler_output']

                    gpt_results = torch.stack(gpt_context_results, dim=1)
            if self.tree_decoder is not None:
                restruct_loss = self.tree_decoder(tree=sample_content_tree)

            logit = self.classifier(results,
                                    default_drop=False if self.myargs.use_dropout else True)
            gpt_logit = None
            if self.myargs.gpt_tree_supervised and self.myargs.use_gpt_tree:
                if self.myargs.use_two_side:
                    toxic_gpt_logit = self.classifier(toxic_gpt_results,
                                                      default_drop=False if self.myargs.use_dropout else True)
                    non_toxic_gpt_logit = self.classifier(non_toxic_gpt_results,
                                                          default_drop=False if self.myargs.use_dropout else True)
                    gpt_logit = (toxic_gpt_logit + non_toxic_gpt_logit) / 2
                else:
                    gpt_logit = self.classifier(gpt_results,
                                                default_drop=False if self.myargs.use_dropout else True)


        # else:
        #     # 将对应prompt位置的最终表示输入到分类器中进行分类
        #     logit = self.classifier(
        #         final_output['last_hidden_state'][torch.arange(0, batchsize).unsqueeze(-1), input_batch['index']])
        else:
            gpt_final_output = self.embed_model(input_ids=input_batch['gpt_context_input_ids'].to(self.device),
                                                token_type_ids=input_batch['gpt_context_token_type_ids'].to(
                                                    self.device),
                                                attention_mask=input_batch['gpt_context_attention_mask'].to(
                                                    self.device))
            results = final_output['last_hidden_state'][
                          torch.arange(0, batchsize).unsqueeze(-1), input_batch['index']] + \
                      gpt_final_output['last_hidden_state'].mean(dim=1, keepdim=True)
            logit = self.classifier(results)

        if analysis:
            if self.myargs.use_tree:
                return sample_content_tree, query_res_attention_scores, query_tree_fea, query_res_feas, tree_after_query
        else:
            return logit, gpt_logit, contrat_loss, toxic_con_loss, toxic_type_con_loss, restruct_loss, cost_distances

    def forward(self, input_batch, task_prompt, analysis=False):
        if analysis:
            if self.myargs.use_tree:
                sample_content_tree, query_res_attention_scores, query_tree_fea, query_res_feas, tree_after_query = self.get_content_out(
                    input_batch=input_batch, task_prompt=task_prompt, analysis=analysis)
                return sample_content_tree, query_res_attention_scores, query_tree_fea, query_res_feas, tree_after_query
        else:
            logit, gpt_logit, contrat_loss, toxic_con_loss, toxic_type_con_loss, restruct_loss, cost_distances = self.get_content_out(
                input_batch=input_batch, task_prompt=task_prompt, analysis=analysis)
            return logit, gpt_logit, contrat_loss, toxic_con_loss, toxic_type_con_loss, restruct_loss, cost_distances
