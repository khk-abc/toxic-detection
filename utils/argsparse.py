import argparse

parser = argparse.ArgumentParser(description='Chinese Toxic Classification')
parser.add_argument('--mode', default="test", type=str, help='train/test')
# 是否微调超参数
parser.add_argument('--tune_param', default=False, type=bool, help='True for param tuning')
parser.add_argument('--tune_samples', default=1, type=int, help='Number of tuning experiments to run')
parser.add_argument('--tune_asha', default=False, type=bool, help='If use ASHA scheduler for early stopping')
parser.add_argument('--tune_file', default='RoBERTa', type=str, help='Suffix of filename for parameter tuning results')
parser.add_argument('--tune_gpu', default=True, type=bool, help='Use GPU to tune parameters')
parser.add_argument("--task", default='joint',type=str, help="choose to predict toxic/non-toxic, hate/offensive, target groups, toxic cato")
parser.add_argument("--emb_model",default='multilingual-bert',type=str,help='choose the embedding model, including bert/roberta/bilstm')
parser.add_argument("--use_local_data",default=False,type=bool,help='whether to use local saved data')
parser.add_argument("--seed",default=42,type=int,help='seed to run')
parser.add_argument("--model_version",default='multitoxic_express',type=str,help='the version of model to make it different')


args = parser.parse_args()

print(args)