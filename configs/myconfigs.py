# coding: UTF-8
import torch


# 2022.7.27 config基类


class Config_base(object):
    """配置参数"""

    def __init__(self, model_name, args=None):
        # path
        self.model_name = model_name

        self.train_path = '../Mydatasets/ToxiCN/data/toxicn_with_facts/train_with_facts_20231227.json'  # 训练集
        self.dev_path = '../Mydatasets/ToxiCN/data/toxicn_with_facts/test_with_facts_20231227.json'  # 验证集
        self.test_path = '../Mydatasets/ToxiCN/data/toxicn_with_facts/test_with_facts_20231227.json'  # 测试集

        self.vocab_path = '../Mydatasets/ToxiCN/data/vocab.pkl'
        self.lexicon_path = '../Mydatasets/ToxiCN/lexicon/'  # 词表

        self.result_path = '../MyToxic-twoTree-gptcontext-content/results/{0}/{1}'.format(args.model_version,
                                                                                                 args.emb_model)
        self.checkpoint_path = '../MyToxic-twoTree-gptcontext-content/results/{0}/{1}'.format(args.model_version,
                                                                                                 args.emb_model)

        self.data_path = self.checkpoint_path

        self.hidden_size = 768

        self.pos_vocab_size=27

        # for attention
        self.num_attention_heads=8
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.3  # v1 0.5
        self.attention_probs_dropout_prob=0.3  # v1 0.5

        self.if_grad = True

        self.weight_decay = 1e-2

        self.NUM = {
            "toxic": 2,
            "toxic_type": 2,
            "expression": 3,
            "target": 5
        }

        self.task_prompt = {
            "toxic": '该文本是否包含毒性内容？',
            "toxic_type": '如果该文本包含毒性内容，所含毒性内容是否针对特定的目标群体？',
            "expression": '如果该文本包含毒性内容并且针对特定目标群体，所含毒性内容采用什么表达方式？',
            "target": '如果该文本包含毒性内容并且针对特定目标群体，所含毒性内容针对哪些群体？'
        }

        self.definitions = [
            '文本包含毒性内容指文本中存在侮辱、谩骂、讽刺等负面消极内容。',
            '目标群体是指文本中毒性内容的针对目标，包括种族群体、性别群体、地区群体、同性恋群体、其他群体。',
            '表达方式是指文本中毒性内容在针对目标时所采用的描述方式，包括直接表达、间接表达和转述引用。',
        ]

        self.definitions_v1 = [
            '文本包含毒性内容是指文本中存在具有恶意、负面或冒犯性的表达，包括但不限于侮辱、谩骂、讽刺、歧视、挑衅、谣言等言辞，可能会伤害或冒犯他人。',
            '目标群体指文本中毒性内容所针对的人群、群体或社会身份，可以由性别、种族、地理位置或其他相关特征而被识别。',
            '表达方式指的是文本中毒性内容在针对目标群体时所采用的阐述方式，包括直接表达、间接表达和转述引用。',
            '直接表达指文本中毒性内容明确使用冒犯性、攻击性或歧视性的语言，直接指向目标群体。',
            '间接表达指文本中毒性内容使用隐喻、比喻、隐晦的方式来传递毒性，使其毒性不易被察觉。',
            '转述引用指文本中毒性内容引用他人的毒性言论或观点，即使作者本身没有采用恶意表达，但仍然传播了毒性内容。'
        ]

        self.num_classes = self.NUM[args.task] if args.task != "joint" else 'None'  # 类别数

        self.pad_size = 80+80
        # self.pad_size = 300

        # model
        self.dropout = 0.5  # 随机失活
        self.vocab_dim = 768
        self.fc_hidden_dim = 256

        self.TASK_NUM = 4

        # train
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')  # 设备
        self.learning_rate = 1e-3 if args.emb_model == 'bilstm' else 1e-5  # 学习率  transformer:5e-4
        self.scheduler = False  # 是否学习率衰减
        self.adversarial = False  # 是否对抗训练
        self.num_warm = 0  # 开始验证的epoch数
        self.num_epochs = 40  # epoch数

        self.batch_size = 32  # mini-batch大小

        # loss
        self.alpha1 = 0.5
        self.gamma1 = 4

        self.toxic_con_loss_weight = 0.2
        self.toxic_type_con_loss_weight = 0.2

        # evaluate
        self.threshold = 0.5  # 二分类阈值
        self.score_key = "F1"  # 评价指标



if __name__=="__main__":
    pass



