import torch.nn as nn
import torch

class TwoLayerFFNNLayer(nn.Module):
    '''
    2-layer FFNN with specified nonlinear function
    must be followed with some kind of prediction layer for actual prediction
    '''

    def __init__(self, config, myargs):
        super(TwoLayerFFNNLayer, self).__init__()
        self.output = config.dropout
        self.input_dim = config.vocab_dim
        self.hidden_dim = config.fc_hidden_dim
        self.dropout = nn.Dropout(config.dropout)
        self.task = myargs.task
        self.config = config

        if myargs.task == 'joint':
            self.out_dim = config.NUM
            self.indexmap = {}
            for i, key in enumerate(config.task_prompt):
                self.indexmap[key] = i  # 输入各项任务prompt的顺序，如toxic:0,toxic_type:1,...
            print(self.indexmap)
            self.model = nn.ModuleDict()
            for key in self.out_dim.keys():
                self.model[key] = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                                nn.Tanh(),
                                                nn.Linear(self.hidden_dim, self.out_dim[key]))
                # self.model[key] = nn.Linear(self.input_dim,self.out_dim[key])

            # self.inint_weight()

        else:
            self.out_dim = config.num_classes
            self.model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(self.hidden_dim, self.out_dim))

    def inint_weight(self):
        from transformers import BertTokenizer, BertModel
        tker = BertTokenizer.from_pretrained(self.config.model_name)
        model = BertModel.from_pretrained(self.config.model_name)
        model.eval()
        for key in self.config.verbor:
            with torch.no_grad():
                inputs = tker(self.config.verbor[key], return_tensors='pt', padding='longest')
                output = model(**inputs)
            self.model[key].weight.data = output['last_hidden_state'].mean(dim=1).detach()  # answer_num,dim
            self.model[key].weight.require_grad = True

    def forward(self, features,default_drop=True):
        if default_drop:
            features = self.dropout(features)
        if self.task == 'joint':
            res_out = {}
            for key in self.out_dim.keys():
                res_out[key] = self.model[key](features[:, self.indexmap[key]])

            return res_out
        else:
            return self.model(features)
