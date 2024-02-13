import sys
import time

from transformers import AutoConfig

sys.path.append('.')
from train_eval.train_eval_for_debug import train, evaluate
import os
import numpy as np
import torch.nn as nn
import argparse
import torch
from ray import tune

from DataLoaders.datasets import MyDataset, MyDataloader
# from DataLoaders.datasets_v2 import MyDataset_v2, MyDataLoader_v2, collen_fn

from utils.utils import get_time_dif

from configs.myconfigs import Config_base
# from mymodels.models import TotalModel
from mymodels_for_debug.models_our_pur import TotalModel
import json
from collections import defaultdict

import pandas as pd

from DataLoaders.bias_compute import compute_bias_score
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser(description='Chinese Toxic Classification')
parser.add_argument('--mode', default="train", type=str, help='train/test')
# 是否微调超参数
parser.add_argument('--tune_param', default=False, type=bool, help='True for param tuning')
parser.add_argument('--tune_samples', default=1, type=int, help='Number of tuning experiments to run')
parser.add_argument('--tune_asha', default=False, type=bool, help='If use ASHA scheduler for early stopping')
parser.add_argument('--tune_file', default='RoBERTa', type=str, help='Suffix of filename for parameter tuning results')
parser.add_argument('--tune_gpu', default=True, type=bool, help='Use GPU to tune parameters')
parser.add_argument("--task", default='joint', type=str,
                    help="choose to predict toxic/non-toxic, hate/offensive, target groups, toxic cato")
parser.add_argument("--emb_model", default='bert', type=str, help='choose the embedding model, including bert/roberta')
parser.add_argument("--seed", default=42, type=int, help='seed to run')
parser.add_argument("--model_version", default='v0', type=str, help='the version of model to make it different')
parser.add_argument("--use_local_data", action='store_true', default=False, help='whether to use local saved data')
parser.add_argument("--device", type=str, default="cuda:3", help='which device')

parser.add_argument("--use_important_score", action='store_true', default=True, help='是否使用重要分数进行剪枝！')
parser.add_argument("--use_extra_share", action='store_false', default=True, help='是否对不同任务的prompt进行共享关联！')
parser.add_argument("--use_tree", action='store_true', default=True, help='是否使用树结构！')
parser.add_argument("--construct_tree_style", default='conv', help='构造树的方法！', choices=['kmeans', 'conv'])
parser.add_argument("--use_decoder", action='store_true', default=False, help='是否使用解码器对语义进行重构！')
parser.add_argument("--use_same_level_contrast", action='store_true', default=False, help='是否根据任务使用对比学习！')
parser.add_argument("--use_toxic_enhance", action='store_true', default=False, help='是否使用毒性词库进行增强！')
parser.add_argument("--splitbatch", action='store_true', default=False, help='是否逐样本构建树！')
parser.add_argument("--notask_in_tree", action='store_true', default=False, help='是否使task对应的token剥离出树的构建！')
parser.add_argument("--enhance_init_style", default='definition', help='当对不同任务的prompt进行共享关联时，采用什么方式初始化：定义增强和正交初始化！'
                    , choices=['definition', 'orthogo'])
parser.add_argument("--enhace_level", default='sent', help='当对不同任务的prompt进行共享关联时，在什么层面进行共享关联！'
                    , choices=['sent', 'token'])
parser.add_argument("--task_prompt_output", default='embeddings_emb', help='对于任务对应的定义以及prompt采用什么方式提取特征！'
                    , choices=['embeddings_emb', 'bert-with-eval'])
parser.add_argument("--use_soft_alpha", type=float, default=0., help='当大于0时使用gpt生成的软标签！')
parser.add_argument("--concat_social_sense", action='store_true', default=False, help='是否串接gpt生成的社会背景含义！')
parser.add_argument("--pad_size", type=int, default=160, help='输入句子长度！')
parser.add_argument("--gpt_pad_size", type=int, default=160, help='gpt背景信息句子长度！')

parser.add_argument("--use_gpt_tree", action='store_true', default=False, help='是否使用gpt构造树以及进一步传输等！')
parser.add_argument("--use_OT", action='store_true', default=False, help='是否使用最优传输！')
parser.add_argument("--use_OT_kernel", action='store_true', default=False, help='是否在传输中使用核函数！')
parser.add_argument("--use_OT_alpha", type=float, default=0.2, help='当使用最优传输时传输的更新因子！')
parser.add_argument("--gpt_tree_supervised", action='store_true', default=False, help='是否针对gpt树查询结果进行任务监督！')
parser.add_argument("--speedup", action='store_true', default=False, help='是否批量传输加快速度！')
parser.add_argument("--detach", action='store_true', default=False, help='加速时是否阶段梯度回传！')
parser.add_argument("--metric", type=str, default='cosine', help='相似度计算指标',choices=['l2','l1','cosine','kl','kl_l2'])
parser.add_argument("--thresh", type=float, default=1e-5, help='传输停止阈值！')
parser.add_argument("--max_iter", type=int, default=100, help='传输停止轮次！')

parser.add_argument("--treequery_version",default='v2',type=str,help='使用tree query的版本！')
parser.add_argument("--delete_pad", default=False, action='store_true', help='在查询时对每一层求和是是否剔除pad字部分！')

parser.add_argument("--learn_path", default=False, action='store_true', help='是否对查询路径进行学习！')

parser.add_argument("--use_two_side", default=False, action='store_true', help='在使用gpt背景信息是否使用双面分析！')

parser.add_argument("--trainable_epislon", default=False, action='store_true', help='传输epsilon是否可训练！')
parser.add_argument("--trainable_Ot_alpha", default=False, action='store_true', help='传输更新因子ot alpha是否可训练！')
parser.add_argument("--learn_ot_alpha", default=False, action='store_true', help='传输更新因子ot alpha是否根据特征训练！')
parser.add_argument("--normal_ot_scores", default=False, action='store_true', help='传输分数是否归一化！')
parser.add_argument("--repeatsent", default=False, action='store_true', help='重复句子平衡长度！')

parser.add_argument("--split_gpt_sample", default=False, action='store_true', help='是否拆开gpt背景和sample的树模块！')
parser.add_argument("--gptall", default=False, action='store_true', help='sample和gpt是否两套tree！')
parser.add_argument("--cat_pooler", default=False, action='store_true', help='传输前是否串接pooler！')

parser.add_argument("--not_clone_tree", default=False, action='store_true', help='不同任务进行查询时是否克隆树，默认克隆！')
parser.add_argument("--data_version", default='no_range', type=str, help='使用gpt背景信息的数据版本！')

parser.add_argument("--train_cost_matrix", default=False, action='store_true', help='是否使用任务相关特征参与代价计算，'
                                                                                    '使代价可训练，且作为特定任务下的条件代价！')
parser.add_argument("--train_cost_matrix_v2", default=False, action='store_true', help='是否使用任务相关特征参与代价计算，'
                                                                                    '使代价可训练，且作为特定任务下的条件代价！')

parser.add_argument("--use_kernel_cost", default=False, action='store_true', help='是否对代价矩阵进行高斯核计算，以便扩散模糊边界！')


parser.add_argument("--use_scheduler", default=False, action='store_true', help='是否学习率更新器！')
parser.add_argument("--add_pooler", default=False, action='store_true', help='是否加上pooler_out！')
parser.add_argument("--task_query_embedding", default=False, action='store_true', help='是否额外学习任务的embedding！')
parser.add_argument("--not_extra_enhance", default=False, action='store_true', help='是否使用问句和定义进行增强，优先级最高！')
parser.add_argument("--use_dropout", default=False, action='store_true', help='是否在分类前使用dropout！')

parser.add_argument("--gpt_loss_alpha", type=float, default=0.1, help='如果使用gpt监督，损失权重！')
parser.add_argument("--tree_dropout", type=float, default=0., help='树的dropout概率，为 0 时表示不使用dropout！')

parser.add_argument("--choose_model", type=str, default='v3', help='模型版本！',)

parser.add_argument("--sample_to_gpt", default=False, action='store_true', help='将样本放到gpt背景中！')
parser.add_argument("--use_attention_transfer", default=False, action='store_true', help='使用注意力传输！')

parser.add_argument("--add_gpt", default=False, action='store_true', help='将样本放到gpt背景中！')

parser.add_argument("--stack_num", type=int, default=2, )
parser.add_argument("--epoch", type=int, default=40, )



args = parser.parse_args()

# TOC
search_space = {
    'learning_rate': tune.choice([1e-5, 5e-5]),
    'num_epochs': tune.choice([5, 10, 15, 20]),
    'batch_size': tune.choice([16, 32, 64]),
    'dropout': tune.choice([0.2, 0.3, 0.5]),
    'seed': tune.choice([42]),
    "pad_size": tune.choice([50, 100])
}


# # THUCnews
# search_space = {
#     'learning_rate': tune.choice([1e-5]),
#     'num_epochs': tune.choice([5]),
#     'batch_size': tune.choice([32]),
#     # 'dropout': tune.choice([0.2, 0.3, 0.5]),
#     'seed': tune.choice([1]),
#     "pad_size" : tune.choice([80]),
#     "alpha1" : tune.choice([0.5])
# }


def convert_label(preds, mapper):
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    final_pred = []
    for pred in preds:
        tmp = []
        if sum(pred) != 0:
            for idx, p in enumerate(pred):
                if p == 1:
                    tmp.append(mapper[idx])
        final_pred.append(tmp)
    return final_pred


if __name__ == '__main__':
    # dataset = 'TOC'  # 数据集
    dataset = "ToxiCN"
    if args.emb_model == 'bert':
        model_name = "./pretrained_models/bert-base-chinese"
    elif args.emb_model == "roberta":
        model_name = "./pretrained_models/chinese-roberta-wwm-ext"
    elif args.emb_model == "hateBERT":
        model_name = "./pretrained_models/hateBERT"

    set_seed(args.seed)

    start_time = time.time()
    print("Loading data...")

    config = Config_base(model_name, args)  # 引入Config参数，包括Config_base和各私有Config

    config.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')  # 设备
    config.pad_size = args.pad_size
    config.num_epochs = args.epoch

    dir = 'seed-{}-{}-maxlen-{}_B-{}-task-{}'.format(args.seed, args.emb_model,
                                                            config.pad_size,
                                                            config.batch_size,
                                                            args.task)
    config.result_path = os.path.join(config.result_path, dir)
    config.checkpoint_path = os.path.join(config.checkpoint_path, dir)
    config.data_path = os.path.join(config.data_path, dir)

    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(os.path.dirname(config.checkpoint_path)):
        os.makedirs(config.checkpoint_path)
    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)

    print(args)

    if args.splitbatch:
        args.update_steps = config.batch_size
        config.batch_size = 1
        print(f'使用splitbatch, batachsize变为{config.batch_size},并且{args.update_steps}更新一次梯度！')


    print(args.use_local_data)
    if not args.use_local_data:
        # config, data_name, task_num=4,add_special_tokens=True, not_test=True
        trn_data = MyDataset(config, args, config.train_path)
        dev_data = MyDataset(config, args, config.dev_path)
        test_data = MyDataset(config, args, config.test_path)
        if not os.path.exists(os.path.dirname(config.data_path)):
            os.makedirs(os.path.dirname(config.data_path))
        torch.save({
            'trn_data': trn_data,
            'dev_data': dev_data,
            'test_data': test_data,
        }, config.data_path + '/data.tar')
    else:
        checkpoint = torch.load(config.data_path + '/data.tar')
        trn_data = checkpoint['trn_data']
        dev_data = checkpoint['dev_data']
        test_data = checkpoint['test_data']
        print('The size of the Training dataset: {}'.format(len(trn_data)))
        print('The size of the Validation dataset: {}'.format(len(dev_data)))
        print('The size of the Test dataset: {}'.format(len(test_data)))
    # data, batch_size, shuffle = True, SEED = 42
    train_iter = MyDataloader(trn_data, batch_size=int(config.batch_size), SEED=args.seed)
    dev_iter = MyDataloader(dev_data, batch_size=int(config.batch_size), shuffle=False)
    test_iter = MyDataloader(test_data, batch_size=int(config.batch_size), shuffle=False)

    # train_iter = MyDataLoader_v2(trn_data.task_prompt, dataset=trn_data, batch_size=int(config.batch_size), shuffle=True, collate_fn=collen_fn)
    # dev_iter = MyDataLoader_v2(dev_data.task_prompt, dataset=dev_data, batch_size=int(config.batch_size), shuffle=False, collate_fn=collen_fn)
    # test_iter = MyDataLoader_v2(test_data.task_prompt, dataset=test_data, batch_size=int(config.batch_size), shuffle=False, collate_fn=collen_fn)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


    if args.data_version=='with_range':
        print('使用带范围提示的gpt背景！')


    def experiment(tune_config):
        if tune_config:
            for param in tune_config:
                setattr(config, param, tune_config[param])
        train(config, args, train_iter, dev_iter, test_iter)


    if args.mode == "train":
        # if args.tune_param:
        #     scheduler = ASHAScheduler(metric='metric', mode="max") if args.tune_asha else None
        #     analysis = tune.run(experiment, num_samples=args.tune_samples, config=search_space,
        #                         resources_per_trial={'gpu': int(args.tune_gpu)},
        #                         scheduler=scheduler,
        #                         verbose=3)
        #
        #     analysis.results_df.to_csv('tune_results_' + args.tune_file + '.csv')
        # # if not tune parameters
        # else:
        #     experiment(tune_config=None)
        params = {}
        params.update(vars(config))
        params.update(vars(args))
        with open(os.path.join(config.result_path,'configs.json'),'w',encoding='utf-8') as f:
            json.dump(params,f,ensure_ascii=False,indent=2)
        experiment(tune_config=None)

    else:

        model = TotalModel(config=config, myargs=args)

        path = './toxic/ToxiCN/MyToxic/saved_dict/sent_query_embeddings_emb_definition_enhance_modeleval_featree/ckp-seed-42-bert-base-chinese-NN_ML-80_D-0.5_B-32_E-40_Lr-1e-05_aplha-0.5-task-joint-BEST.tar'

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        model = model.to(config.device)
        print(next(model.named_parameters())[1].device)

        loss_fn = nn.BCEWithLogitsLoss()
        dev_scores, preds, bias_logit = evaluate(config, args, model, loss_fn, test_iter, data_name='TEST',
                                                 compute_bias=True)

        print(dev_scores)

        mapper = json.load(open('./toxic/ToxiCN/ToxiCN_ex/ToxiCN/data/mapper.json', 'r'))
        reverse_mapper = defaultdict(dict)

        results = []
        for task in mapper:
            for key, value in mapper[task].items():
                reverse_mapper[task][value] = key
            predictions = []
            for pred in preds:
                predictions.extend(pred[task])
            print(convert_label(preds=predictions, mapper=reverse_mapper[task]))
            results.append(convert_label(preds=predictions, mapper=reverse_mapper[task]))

        pd_view = pd.DataFrame([[','.join(toxic), ','.join(toxic_type), ','.join(expression), ','.join(target)]
                                for toxic, toxic_type, expression, target in zip(*results)],
                               columns=['toxic', 'toxic_type', 'expression', 'target'])
        print(pd_view)

        print((pd_view.loc[1, 'toxic']))
        pd_toxic = pd_view[pd_view['toxic'] == 'toxic']
        pd_non_toxic = pd_view[pd_view['toxic'] == 'non-toxic']

        pd_toxic_with_all = pd_toxic.query('toxic_type !="" & expression!="" & target !=""')
        pd_toxic_with_toxic_type = pd_toxic.query('toxic_type!=""')
        pd_toxic_with_expression = pd_toxic.query('expression!="" & toxic_type=="hate"')
        pd_toxic_with_target = pd_toxic.query('target !="" & toxic_type=="hate"')

        pd_non_toxic_no_all = pd_non_toxic.query('toxic_type=="" & expression=="" & target ==""')
        pd_non_toxic_no_toxic_type = pd_non_toxic.query('toxic_type==""')
        pd_non_toxic_no_expression = pd_non_toxic.query('expression==""')
        pd_non_toxic_no_target = pd_non_toxic.query('target ==""')

        pd_offensive_toxic = pd_toxic[pd_toxic['toxic_type'] == 'offensive']
        pd_hate_toxic = pd_toxic[pd_toxic['toxic_type'] == 'hate']

        pd_offensive_toxic_noexpression = pd_offensive_toxic[pd_offensive_toxic['expression'] == '']
        pd_offensive_toxic_notarget = pd_offensive_toxic[pd_offensive_toxic['target'] == '']
        pd_offensive_toxic_noall = pd_offensive_toxic.query('expression=="" & target ==""')

        pd_hate_toxic_noexpression = pd_hate_toxic[pd_hate_toxic['expression'] == '']
        pd_hate_toxic_notarget = pd_hate_toxic[pd_hate_toxic['target'] == '']

        print('non-toxic:\n', pd_non_toxic)
        print('non-toxic_no_all:\n', pd_non_toxic_no_all)

        print('pd_offensive_toxic:\n', pd_offensive_toxic)
        print('pd_hate_toxic:\n', pd_hate_toxic)
        print('pd_offensive_toxic_noexpression:\n', pd_offensive_toxic_noexpression)
        print('pd_offensive_toxic_notarget:\n', pd_offensive_toxic_notarget)

        print('toxic_num:', len(pd_toxic))
        print('toxic_with_toxic_type_num:', len(pd_toxic_with_toxic_type))
        print('toxic_with_expression_num:', len(pd_toxic_with_expression))
        print('toxic_with_target_num:', len(pd_toxic_with_target))
        print('toxic_with_all_num:', len(pd_toxic_with_all))

        print('non-toxic_num:', len(pd_non_toxic))
        print('non-toxic_no_toxic_type_num:', len(pd_non_toxic_no_toxic_type))
        print('non-toxic_no_expression_num:', len(pd_non_toxic_no_expression))
        print('non-toxic_no_target_num:', len(pd_non_toxic_no_target))
        print('non-toxic_no_all_num:', len(pd_non_toxic_no_all))

        print('pd_offensive_toxic_num:', len(pd_offensive_toxic))
        print('pd_offensive_toxic_noexpression_num:', len(pd_offensive_toxic_noexpression))
        print('pd_offensive_toxic_notarget_num:', len(pd_offensive_toxic_notarget))
        print('pd_offensive_toxic_noall_num:', len(pd_offensive_toxic_noall))

        print('pd_hate_toxic_num:', len(pd_hate_toxic))
        print('pd_hate_toxic_noexpression_num:', len(pd_hate_toxic_noexpression))
        print('pd_hate_toxic_notarget_num:', len(pd_hate_toxic_notarget))

        pd_view.to_csv('./toxic/ToxiCN/MyToxic/predisions.csv',
                       sep='\t')

        ## for computing bias
        bias_logit = torch.concat(bias_logit, dim=0)
        print(bias_logit.shape)

        compute_bias_score(predict_label=[i[0] for i in results[0]], predict_logit=bias_logit.numpy().tolist())
