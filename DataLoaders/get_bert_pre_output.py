

from transformers import BertTokenizer, BertModel
import torch

task_prompt = {
    "toxic": '该文本是否包含毒性内容？',
    "toxic_type": '如果该文本包含毒性内容，所含毒性内容是否针对特定的目标群体？',
    "expression": '如果该文本包含毒性内容并且针对特定目标群体，所含毒性内容采用什么表达方式？',
    "target": '如果该文本包含毒性内容并且针对特定目标群体，所含毒性内容针对哪些群体？'
}

definitions = [
    '文本包含毒性内容指文本中存在侮辱、谩骂、讽刺等负面消极内容。',
    '目标群体是指文本中毒性内容的针对目标，包括种族群体、性别群体、地区群体、同性恋群体、其他群体。',
    '表达方式是指文本中毒性内容在针对目标时所采用的描述方式，包括直接表达、间接表达和转述引用。',
]

definitions_v1 = [
            '文本包含毒性内容是指文本中存在具有恶意、负面或冒犯性的表达，包括但不限于侮辱、谩骂、讽刺、歧视、挑衅、谣言等言辞，可能会伤害或冒犯他人。',
            '目标群体指文本中毒性内容所针对的人群、群体或社会身份，可以由性别、种族、地理位置或其他相关特征而被识别。',
            '如果文本没有毒性，则文本没有特定的目标群体。'
            '表达方式指的是文本中毒性内容在针对目标群体时所采用的阐述方式，包括直接表达、间接表达和转述引用。',
            '直接表达指文本中毒性内容明确使用冒犯性、攻击性或歧视性的语言，直接指向目标群体。',
            '间接表达指文本中毒性内容使用隐喻、比喻、隐晦的方式来传递毒性，使其毒性不易被察觉。',
            '转述引用指文本中毒性内容引用他人的毒性言论或观点，即使作者本身没有采用恶意表达，但仍然传播了毒性内容。'
        ]

tker = BertTokenizer.from_pretrained('./pretrained_models/bert-base-chinese')
model = BertModel.from_pretrained('./pretrained_models/bert-base-chinese')

tker_roberta = BertTokenizer.from_pretrained('./pretrained_models/chinese-roberta-wwm-ext')
model_roberta = BertModel.from_pretrained('./pretrained_models/chinese-roberta-wwm-ext')



tker_hatebert = BertTokenizer.from_pretrained('./pretrained_models/hateBERT')
model_hatebert = BertModel.from_pretrained('./pretrained_models/hateBERT')

model.eval()
model_roberta.eval()
model_hatebert.eval()

with torch.no_grad():
    # # definitions
    # # # bert
    # # inputs_definitions = tker(definitions, return_tensors='pt', padding='longest')
    # #
    # # output_definitions = model(**inputs_definitions)
    # # output_definitions_embeddings = model.embeddings(inputs_definitions['input_ids'])
    # #
    # # output_definitions['last_hidden_state'] = output_definitions['last_hidden_state'] * inputs_definitions['attention_mask'].unsqueeze(-1)
    # # output_definitions_embeddings = output_definitions_embeddings * inputs_definitions['attention_mask'].unsqueeze(-1)
    # #
    # # torch.save(output_definitions,'./definitions.ckpt')
    # # torch.save(output_definitions_embeddings,'./definitions_from_embeddings.ckpt')
    #
    # # roberta
    # inputs_definitions = tker_roberta(definitions, return_tensors='pt', padding='longest')
    #
    #
    # output_definitions = model_roberta(**inputs_definitions)
    # output_definitions_embeddings = model_roberta.embeddings(inputs_definitions['input_ids'])
    #
    # output_definitions['last_hidden_state'] = output_definitions['last_hidden_state'] * inputs_definitions[
    #     'attention_mask'].unsqueeze(-1)
    # output_definitions_embeddings = output_definitions_embeddings * inputs_definitions['attention_mask'].unsqueeze(-1)
    #
    # torch.save(output_definitions, './definitions_roberta.ckpt')
    # torch.save(output_definitions_embeddings, './definitions_from_embeddings_roberta.ckpt')
    #
    #
    # # definitions_V1
    # # # bert
    # # inputs_definitions_v1 = tker(definitions_v1, return_tensors='pt', padding='longest')
    # #
    # # output_definitions_v1 = model(**inputs_definitions_v1)
    # # output_definitions_embeddings_v1 = model.embeddings(inputs_definitions_v1['input_ids'])
    # #
    # # output_definitions_v1['last_hidden_state'] = output_definitions_v1['last_hidden_state'] * inputs_definitions_v1['attention_mask'].unsqueeze(-1)
    # # output_definitions_embeddings_v1 = output_definitions_embeddings_v1 * inputs_definitions_v1['attention_mask'].unsqueeze(-1)
    # #
    # # torch.save(output_definitions_v1,'./definitions_v1.ckpt')
    # # torch.save(output_definitions_embeddings_v1,'./definitions_from_embeddings_v1.ckpt')
    #
    # roberta
    inputs_definitions_v1 = tker_roberta(definitions_v1, return_tensors='pt', padding='longest')

    output_definitions_v1 = model_roberta(**inputs_definitions_v1)
    output_definitions_embeddings_v1 = model_roberta.embeddings(inputs_definitions_v1['input_ids'])

    output_definitions_v1['last_hidden_state'] = output_definitions_v1['last_hidden_state'] * inputs_definitions_v1['attention_mask'].unsqueeze(-1)
    output_definitions_embeddings_v1 = output_definitions_embeddings_v1 * inputs_definitions_v1['attention_mask'].unsqueeze(-1)

    torch.save(output_definitions_v1,'./definitions_v1_roberta.ckpt')
    torch.save(output_definitions_embeddings_v1,'./definitions_from_embeddings_v1_roberta.ckpt')
    #
    #
    # # # task_prompt
    # # inputs_task_prompt = tker(list(task_prompt.values()), return_tensors='pt', padding='longest')
    # #
    # # output_task_prompt = model(**inputs_task_prompt)
    # # output_task_prompt_embeddings = model.embeddings(inputs_task_prompt['input_ids'])
    # #
    # # torch.save(output_task_prompt,'./task_prompt.ckpt')
    # # torch.save(output_task_prompt_embeddings,'./task_prompt_from_embeddings.ckpt')

    # hatebert
    inputs_definitions = tker_hatebert(definitions_v1, return_tensors='pt', padding='longest')

    output_definitions = model_hatebert(**inputs_definitions)
    output_definitions_embeddings = model_hatebert.embeddings(inputs_definitions['input_ids'])

    output_definitions['last_hidden_state'] = output_definitions['last_hidden_state'] * inputs_definitions[
        'attention_mask'].unsqueeze(-1)
    output_definitions_embeddings = output_definitions_embeddings * inputs_definitions['attention_mask'].unsqueeze(-1)

    torch.save(output_definitions, './definitions_v1_hatebert.ckpt')
    torch.save(output_definitions_embeddings, './definitions_v1_from_embeddings_hatebert.ckpt')


tmp = torch.load('./definitions_v1_hatebert.ckpt')
tmp1 = torch.load('./definitions_v1_from_embeddings_hatebert.ckpt')

print(tmp['last_hidden_state'].shape)
print(tmp1.shape)
print('last_hidden_state' in tmp)

