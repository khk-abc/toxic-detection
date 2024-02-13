import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import time
import json
import copy
import sys
sys.path.append('..')
from DataLoaders.datasets import to_tensor
from DataLoaders.utils import convert_onehot
from utils.utils import get_time_dif

from utils.utils import convert_onehot_to_onelabel


# from mymodels_for_debug.models_our_pur import TotalModel
# from mymodels_for_debug.models_our_pur_two_side_gpt import TotalModel
# from mymodels_for_debug.models_our_pur_two_side_gpt_v2 import TotalModel

def choose_model(version='v2'):
    if version=='ours_ot_debug':
        from mymodels_for_debug.ours_ot_debug import TotalModel
        print('选择模型ours_ot_debug')
        return TotalModel
    else:
        raise ValueError('Wrong model version!')




def train(config, myargs, train_iter, dev_iter, test_iter):
    temp_name = config.model_name.split('/')[-1]
    model_name = 'seed-{}-task-{}'.format(myargs.seed, myargs.task)
    print(model_name)
    TotalModel = choose_model(version=myargs.choose_model)

    model = TotalModel(config=config, myargs=myargs).to(config.device)
    embed_optimizer = optim.AdamW(model.embed_model.parameters(), lr=1e-5,
    # embed_optimizer = optim.AdamW(model.embed_model.parameters(), lr=1e-6,
                                  weight_decay=config.weight_decay)
    classifier_optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-5,
    # classifier_optimizer = optim.AdamW(model.classifier.parameters(), lr=3e-5,
                                       weight_decay=config.weight_decay)


    others_optimizer = optim.AdamW([p for n,p in model.named_parameters()
                                    if "embed_model" not in n and "classifier" not in n],
                                   lr=5e-5,
                                   # lr=1e-4,
                                   weight_decay=config.weight_decay
                                   )

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(others_optimizer, mode='max', patience=5,
    #                                                  threshold=1e-4, threshold_mode='abs',
    #                                                  min_lr=8e-6, factor=0.9)

    if not myargs.use_scheduler:
        print("不使用scheduler！")
        scheduler = None
    else:
        print("使用scheduler！")

    # fgm = FGM(embed_model, epsilon=1, emb_name='word_embeddings.')
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = get_loss_func("FL", [0.4, 0.6], config.num_classes, config.alpha1)
    max_score = 0

    for epoch in range(config.num_epochs):
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))
        loss_all = 0.
        preds = []
        labels = []

        for step,batch in enumerate(tqdm(train_iter, desc='Training', colour='MAGENTA')):
            if not myargs.splitbatch:
                model.zero_grad()
            input_batch = to_tensor(batch)
            # input_batch = batch
            logit,gpt_logit, contrat_loss, toxic_con_loss,toxic_type_con_loss,restruct_loss,cost_distances = model(input_batch,task_prompt=train_iter.task_prompt)

            if myargs.task == 'joint':
                for key in logit:
                    logit[key] = logit[key].cpu()
                    if gpt_logit is not None:
                        gpt_logit[key] = gpt_logit[key].cpu()
            else:
                logit = logit.cpu()
                if gpt_logit is not None:
                    gpt_logit = gpt_logit.cpu()

            if myargs.task == "toxic":
                label = input_batch['toxic']
            elif myargs.task == "toxic_type":
                label = input_batch['toxic_type']
            elif myargs.task == "expression":
                label = input_batch['expression']
            elif myargs.task == "target":
                label = input_batch['target']
            elif myargs.task == "joint":
                label = {"toxic": input_batch['toxic'],
                         'toxic_type': input_batch['toxic_type'],
                         'expression': input_batch['expression'],
                         'target': input_batch['target']}
            else:
                raise ValueError("Wrong task! task should be toxic/toxic_type/expression/target")
            if myargs.task == 'joint':
                loss = 0.
                for key in label.keys():
                    loss += loss_fn(logit[key], label[key].float())
                    if gpt_logit is not None:
                        loss += myargs.gpt_loss_alpha*loss_fn(gpt_logit[key], label[key].float())

                if myargs.use_soft_alpha>0:
                    loss += myargs.use_soft_alpha*F.binary_cross_entropy_with_logits(input=logit['toxic'],target=input_batch['toxic_soft'])
                    loss += myargs.use_soft_alpha*F.binary_cross_entropy_with_logits(input=logit['toxic_type'],target=input_batch['toxic_type_soft'])
                    loss += myargs.use_soft_alpha*F.binary_cross_entropy_with_logits(input=logit['expression'],target=input_batch['expression_soft'])
                    loss += myargs.use_soft_alpha*F.binary_cross_entropy_with_logits(input=logit['target'],target=input_batch['target_soft'])
            else:
                loss = loss_fn(logit, label.float())

            if myargs.task == "toxic":  # binary classification
                pred = get_preds(num_classes=config.NUM[myargs.task], logit=logit)
                preds.extend(pred)
            elif myargs.task == "expression":
                pred = get_preds_task2_4(num_classes=config.NUM[myargs.task], logit=logit)
                preds.extend(pred)
            elif myargs.task == "toxic_type":  # multi class classification
                pred = get_preds_task2_4(num_classes=config.NUM[myargs.task], logit=logit)
                preds.extend(pred)
            elif myargs.task == "target":  # multi-label classification
                pred = get_preds_task3(num_classes=config.NUM[myargs.task], logit=logit)
                preds.extend(pred)
            elif myargs.task == "joint":
                pred = {
                    "toxic": get_preds(num_classes=config.NUM['toxic'], logit=logit['toxic']),
                    "toxic_type": get_preds_task2_4(num_classes=config.NUM['toxic_type'], logit=logit['toxic_type']),
                    'expression': get_preds_task2_4(num_classes=config.NUM['expression'], logit=logit['expression']),
                    'target': get_preds_task3(num_classes=config.NUM['target'], logit=logit['target'])
                }
                preds.append(pred)

            if myargs.task == 'joint':
                new_label = {}
                for key in label.keys():
                    new_label[key] = label[key].detach().numpy()
                labels.append(new_label)
            else:
                labels.extend(label.detach().numpy())

            if toxic_con_loss is not None:
                loss += config.toxic_con_loss_weight*toxic_con_loss.cpu()
            if toxic_type_con_loss is not None:
                loss += config.toxic_type_con_loss_weight*toxic_type_con_loss.cpu()

            if contrat_loss is not None:
                loss += contrat_loss.cpu()

            if restruct_loss is not None:
                print('raw loss:', loss)
                print('restruct loss:', restruct_loss)
                loss += restruct_loss.cpu()

            if cost_distances is not None:
                # print(cost_distances)
                loss += 10*cost_distances.cpu()

            loss_all += loss.item()


            if myargs.splitbatch:
                if (step+1)%myargs.update_steps==0:
                    loss.backward()

                    embed_optimizer.step()
                    classifier_optimizer.step()
                    others_optimizer.step()
                    embed_optimizer.zero_grad()
                    classifier_optimizer.zero_grad()
                    others_optimizer.zero_grad()
                    model.zero_grad()
                else:
                    loss.backward()
            else:
                embed_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                others_optimizer.zero_grad()
                loss.backward()

                embed_optimizer.step()
                classifier_optimizer.step()
                others_optimizer.step()

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time) / 60.))
        print("TRAINED for {} epochs".format(epoch))

        # 验证
        if epoch >= config.num_warm:
            if myargs.task == 'joint':
                # print("training loss: loss={}".format(loss_all/len(data)))
                trn_scores = {}
                # dev_scores = {}
                for key in ['toxic', 'toxic_type', 'expression', 'target']:
                    pred_tmp = []
                    label_tmp = []
                    for ps, ls in zip(preds, labels):
                        pred_tmp.extend(ps[key])
                        label_tmp.extend(ls[key])
                    trn_scores[key] = get_scores(pred_tmp, label_tmp, loss_all, len(train_iter), data_name="TRAIN")
                dev_scores, _ = evaluate(config, myargs, model, loss_fn, dev_iter, data_name='DEV')

                if scheduler is not None:
                    total_f1_scores = 0.
                    for key in list(dev_scores.keys()):
                        total_f1_scores += dev_scores[key]['F1']
                    scheduler.step(total_f1_scores)

                model_name = model_name.split("/")[-1]

                if not os.path.exists(config.result_path):
                    os.makedirs(config.result_path)

                f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
                f.write(
                    ' ==================================================  Epoch: {}  ==================================================\n'.format(
                        epoch))
                f.write('TrainScore: \n{}\nEvalScore: \n{}\n'.format(json.dumps(trn_scores), json.dumps(dev_scores)))
                print(dev_scores)
                max_score = save_best(config, epoch, model_name, model, dev_scores, max_score, myargs)
            else:
                # print("training loss: loss={}".format(loss_all/len(data)))
                trn_scores = get_scores(preds, labels, loss_all, len(train_iter), data_name="TRAIN")
                dev_scores, _ = evaluate(config, myargs, model, loss_fn, dev_iter, data_name='DEV')

                model_name = model_name.split("/")[-1]

                if not os.path.exists(config.result_path):
                    os.makedirs(config.result_path)

                f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
                f.write(
                    ' ==================================================  Epoch: {}  ==================================================\n'.format(
                        epoch))
                f.write('TrainScore: \n{}\nEvalScore: \n{}\n'.format(json.dumps(trn_scores), json.dumps(dev_scores)))
                max_score = save_best(config, epoch, model_name, model, dev_scores, max_score, myargs)
        print("ALLTRAINED for {} epochs".format(epoch))

    path = '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    test_scores, _ = evaluate(config, myargs, model, loss_fn, test_iter, data_name='TEST')
    print('TEST:', test_scores)
    f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
    f.write('Test: \n{}\n'.format(json.dumps(test_scores)))


def evaluate(config, myargs, model, loss_fn, dev_iter, data_name='DEV',compute_bias=False):
    model.eval()
    loss_all = 0.
    preds = []
    labels = []

    bias_logits =  []
    for batch in tqdm(dev_iter, desc='Evaling', colour='CYAN'):
        with torch.no_grad():
            input_batch = to_tensor(batch)
            # input_batch = batch
            logit, gpt_logit, contrat_loss, toxic_con_loss, toxic_type_con_loss, restruct_loss,cost_distances = model(input_batch,dev_iter.task_prompt)

            if myargs.task == 'joint':
                for key in logit:
                    logit[key] = logit[key].cpu()
                    if gpt_logit is not None:
                        gpt_logit[key] = gpt_logit[key].cpu()
            else:
                logit = logit.cpu()
                if gpt_logit is not None:
                    gpt_logit = gpt_logit.cpu()


            if compute_bias:
                bias_logits.append(logit['toxic'])


            if myargs.task == "toxic":
                label = input_batch['toxic']
            elif myargs.task == "toxic_type":
                label = input_batch['toxic_type']
            elif myargs.task == "expression":
                label = input_batch['expression']
            elif myargs.task == "target":
                label = input_batch['target']
            elif myargs.task == "joint":
                label = {"toxic": input_batch['toxic'],
                         'toxic_type': input_batch['toxic_type'],
                         'expression': input_batch['expression'],
                         'target': input_batch['target']}
            else:
                raise ValueError("Wrong task! task should be toxic/toxic_type/expression/target")

            if myargs.task == 'joint':
                loss = 0.
                for key in label.keys():
                    loss += loss_fn(logit[key], label[key].float())
                    if gpt_logit is not None:
                        loss += 0.1*loss_fn(gpt_logit[key], label[key].float())
            else:
                loss = loss_fn(logit, label.float())

            if myargs.task == "toxic":  # binary classification
                pred = get_preds(num_classes=config.NUM[myargs.task], logit=logit)
                preds.extend(pred)
            elif myargs.task == "expression" or myargs.task == "toxic_type":  # multi class classification
                pred = get_preds_task2_4(num_classes=config.NUM[myargs.task], logit=logit)
                preds.extend(pred)
            elif myargs.task == "target":  # multi-label classification
                pred = get_preds_task3(num_classes=config.NUM[myargs.task], logit=logit)
                preds.extend(pred)
            elif myargs.task == "joint":
                pred = {
                    "toxic": get_preds(num_classes=config.NUM["toxic"], logit=logit['toxic']),
                    "toxic_type": get_preds_task2_4(num_classes=config.NUM["toxic_type"], logit=logit['toxic_type']),
                    'expression': get_preds_task2_4(num_classes=config.NUM['expression'], logit=logit['expression']),
                    'target': get_preds_task3(num_classes=config.NUM['target'], logit=logit['target'])
                }
                preds.append(pred)

            if myargs.task == 'joint':
                new_label = {}
                for key in label.keys():
                    new_label[key] = label[key].detach().numpy()
                labels.append(new_label)
            else:
                labels.extend(label.detach().numpy())

            # if toxic_con_loss is not None:
            #     loss += config.toxic_con_loss_weight*toxic_con_loss.cpu()
            # if toxic_type_con_loss is not None:
            #     loss += config.toxic_type_con_loss_weight*toxic_type_con_loss.cpu()

            loss_all += loss.item()

    if myargs.task == 'joint':
        dev_scores = {}
        for key in ['toxic', 'toxic_type', 'expression', 'target']:
            pred_tmp = []
            label_tmp = []
            for ps, ls in zip(preds, labels):
                # print(ps)
                # print(ls)
                pred_tmp.extend(ps[key])
                label_tmp.extend(ls[key])
            # print("preds:",preds)
            # print("labels:",labels)
            dev_scores[key] = get_scores(pred_tmp, label_tmp, loss_all, len(dev_iter), data_name=data_name)
    else:
        dev_scores = get_scores(preds, labels, loss_all, len(dev_iter), data_name=data_name)
    # if data_name != "TEST": # 2022.9.20 命令行输入为test模式时，不调用tune
    # tune.report(metric=dev_scores)

    if compute_bias:
        return dev_scores, preds,bias_logits

    else:
        return dev_scores, preds


# For Multi Classfication
def get_preds(num_classes, logit, style='sigmoid'):
    if style=='sigmoid':
        results = torch.max(logit.data, 1)[1].cpu().numpy()
        new_results = []
        for result in results:
            result = convert_onehot(num_classes, result)
            new_results.append(result)
        return new_results
    elif style=='softmax':
        results = torch.max(logit.data, 1)[1].cpu().numpy()
        new_results = []
        for result in results:
            result = convert_onehot(num_classes, result)
            new_results.append(result)
        return new_results


# Task 2 and 4: 多分类 Toxic Type Discrimination and d Expression Type Detection
def get_preds_task2_4(num_classes, logit, style='sigmoid'):
    if style=='sigmoid':
        all_results = []
        logit_ = torch.sigmoid(logit)
        results_pred = torch.max(logit_.data, 1)[0].cpu().numpy()
        results = torch.max(logit_.data, 1)[1].cpu().numpy()  # index for maximum probability
        for i in range(len(results)):
            if results_pred[i] < 0.5:
                result = [0 for _ in range(num_classes)]
            else:
                result = convert_onehot(num_classes, results[i])
            all_results.append(result)
        return all_results

    elif style=='softmax':
        all_results = []
        logit_ = torch.softmax(logit)
        results_pred = torch.max(logit_.data, 1)[0].cpu().numpy()
        results = torch.max(logit_.data, 1)[1].cpu().numpy()  # index for maximum probability
        for i in range(len(results)):
            if results_pred[i] < 0.5:
                result = [0 for _ in range(num_classes)]
            else:
                result = convert_onehot(num_classes, results[i])
            all_results.append(result)
        return all_results


# Task 3: 多标签分类 Targeted Group Detection
def get_preds_task3(num_classes, logit):
    all_results = []
    logit_ = torch.sigmoid(logit)
    results_pred = torch.max(logit_.data, 1)[0].cpu().numpy()
    results = torch.max(logit_.data, 1)[1].cpu().numpy()
    logit_ = logit_.detach().cpu().numpy()
    for i in range(len(results)):
        if results_pred[i] < 0.5:
            result = [0 for _ in range(num_classes)]
        else:
            result = get_pred_task3(logit_[i])
        all_results.append(result)
    return all_results


def get_pred_task3(logit):
    result = [0 for i in range(len(logit))]
    for i in range(len(logit)):
        if logit[i] >= 0.5:
            result[i] = 1
    return result


def get_scores(all_preds, all_lebels, loss_all, len, data_name):
    score_dict = dict()
    f1 = f1_score(y_pred=all_preds, y_true=all_lebels, average='weighted')
    acc = accuracy_score(y_pred=all_preds, y_true=all_lebels)
    all_f1 = f1_score(y_pred=all_preds, y_true=all_lebels, average=None)
    pre = precision_score(y_pred=all_preds, y_true=all_lebels, average='weighted')
    recall = recall_score(y_pred=all_preds, y_true=all_lebels, average='weighted')

    try:
        confu_mat = confusion_matrix(y_true=[convert_onehot_to_onelabel(l.tolist()) for l in all_lebels],
                                     y_pred=[convert_onehot_to_onelabel(p) for p in all_preds]).tolist()
    except:
        # print(all_preds)
        # print(all_lebels)
        confu_mat = None

    score_dict['F1'] = f1
    score_dict['accuracy'] = acc
    score_dict['all_f1'] = all_f1.tolist()
    score_dict['precision'] = pre
    score_dict['recall'] = recall
    score_dict['confuse_matrix'] = confu_mat

    score_dict['all_loss'] = loss_all / len
    print("Evaling on \"{}\" data".format(data_name))
    # for s_name, s_val in score_dict.items():
    #     print("{}: {}".format(s_name, s_val))
    return score_dict


def save_best(config, epoch, model_name, model, score, max_score, myargs):
    score_key = config.score_key

    if myargs.task == 'joint':
        curr_score = score['expression'][score_key] + score['target'][score_key] + score['toxic_type'][score_key] + \
                     score['toxic'][score_key]
    else:
        curr_score = score[score_key]

    print('The epoch_{} {}: {}\nCurrent max {}: {}'.format(epoch, score_key, curr_score, score_key, max_score))

    if curr_score > max_score or epoch == 0:
        torch.save(model.state_dict(), '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST'))
        return curr_score
    else:
        return max_score
