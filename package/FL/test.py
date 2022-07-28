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
        correct  = torch.tensor([0.0] * 10)
        # 各種圖的數量
        gold_all = torch.tensor([0.0] * 10)
    else:
        print("Unknown dataset")
        exit(0)

    # 攻擊效果
    poison_correct = 0.0

    data_loader = DataLoader(datatest, batch_size=f.test_bs)
    
    print(' test data_loader(per batch size):',len(data_loader))
    
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
            
            # 預測和正解相同
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1
            elif f.attack_mode == 'poison':
                # 被攻擊的目標，攻擊效果如何
                for label in f.target_label:
                    if int(y_pred[pred_idx]) != label and int(y_gold[pred_idx]) == label:
                        poison_correct += 1


    test_loss /= len(data_loader.dataset)

    accuracy = (sum(correct) / sum(gold_all)).item()
    
    acc_per_label = correct / gold_all

    poison_acc = 0

    if(f.attack_mode == 'poison'):
        tmp = 0
        for label in f.target_label:
            tmp += gold_all[label].item()
        poison_acc = poison_correct/tmp
    
    return accuracy, test_loss, acc_per_label.tolist(), poison_acc





