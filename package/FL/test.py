'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/test.py
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..config import for_FL as f

f.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')

def test_img_poison(net, datatest):

    net.eval()
    test_loss = 0
    if f.dataset == "mnist":
        # 各種圖預測正確的數量
        # SEPERATE INTO TWO CASE: 1. normal dataset(without poison) 2. poison dataset(all poison)
        correct  = torch.tensor([0.0] * 10)
        correct_pos = torch.tensor([0.0] * 10)
        # 各種圖的數量
        gold_all = torch.tensor([0.0] * 10)
        gold_all_pos = torch.tensor([0.0] * 10)
    else:
        print("Unknown dataset")
        exit(0)

    # 攻擊效果
    poison_correct = 0.0

    data_loader = DataLoader(datatest, batch_size=f.test_bs)
    
    print(' test data_loader(per batch size):',len(data_loader))
    
    # FIRST TEST: normal dataset
    for idx, (data, target) in enumerate(data_loader):
        if f.gpu != -1:
            data, target = data.to(f.device), target.to(f.device)

        log_probs = net(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # 預測解
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # 正解
        y_gold = target.data.view_as(y_pred).squeeze(1)
        
        y_pred = y_pred.squeeze(1)


        for pred_idx in range(len(y_pred)):
            gold_all[ y_gold[pred_idx] ] += 1
            # ACCURACY RATE
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1

    # SECOND TEST: poison dataset
    for idx, (data, target) in enumerate(data_loader):
        if f.gpu != -1:
            data, target = data.to(f.device), target.to(f.device)

        # count = 1 # for TEST
        for label_idx in range(len(target)):
            target[label_idx] = f.error_label

            data[label_idx][0][27][25] = 2
            data[label_idx][0][27][27] = 2
            data[label_idx][0][25][25] = 2
            data[label_idx][0][25][27] = 2
            # # CHECK IMAGE
            # plt.imshow(images[i][0], cmap='gray')
            # name = "file" + str(count) + ".png"
            # print(name)
            # plt.savefig(name)
            # plt.close()
            # count += 1

        log_probs_pos = net(data)
        test_loss += F.cross_entropy(log_probs_pos, target, reduction='sum').item()
        # 預測解
        y_pred_pos = log_probs_pos.data.max(1, keepdim=True)[1]
        # 正解
        y_gold_pos = target.data.view_as(y_pred_pos).squeeze(1)
        
        y_pred_pos = y_pred_pos.squeeze(1)

        for pred_idx in range(len(y_pred_pos)):
            gold_all_pos[ y_gold_pos[pred_idx] ] += 1
            # POISON ATTACK SUCCESS RATE
            if y_pred_pos[pred_idx] == y_gold_pos[pred_idx]:
                correct_pos[y_pred_pos[pred_idx]] += 1



    test_loss /= len(data_loader.dataset) * 2

    accuracy = (sum(correct) / sum(gold_all)).item()
    
    acc_per_label = correct / gold_all

    poison_acc = 0

    accuracy_all = ((sum(correct) + sum(correct_pos)) / (sum(gold_all) + sum(gold_all_pos))).item()

    if(f.attack_mode == 'poison'):
        poison_acc = (sum(correct_pos) / sum(gold_all_pos)).item()
    
    return accuracy, test_loss, acc_per_label.tolist(), poison_acc, accuracy_all





