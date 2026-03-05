import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import  datetime

import enum
import re
# from symbol import testlist_star_expr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
import os
import pandas as pd
from sklearn.model_selection import KFold

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset
# import redis
import pickle
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    roc_auc_score, roc_curve
import random
import torch.backends.cudnn as cudnn
import json
import joblib
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from Opt.lookahead import Lookahead
from Opt.radam import RAdam
from torch.cuda.amp import GradScaler, autocast
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import h5py



# ✅ 自定义 PyTorch Dataset
class FeatureDataset(Dataset):
    def __init__(self, df, label_col='label'):
        self.labels = df[label_col].values.astype(np.int64)
        self.features = df.drop(columns=['filename', label_col]).values.astype(np.float32)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label1 = self.labels[idx]
        label = np.zeros(2)
        if int(label1) <= (len(label) - 1):
            label[int(label1)] = 1
        label = torch.tensor(np.array(label))

        return torch.tensor(self.features[idx]), torch.tensor(label)


def train(train_df, milnet, criterion, optimizer, args, log_path):
    milnet.train()
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0
    scaler = GradScaler()
    # test_labels = []
    test_predictions = []
    # for epoch in range(args.num_epochs):
    #     milnet.train()
    for i, (features, labels) in enumerate(train_df):
        optimizer.zero_grad()
        outputs = milnet(features)
        # loss = criterion(outputs, labels)
        loss = criterion(outputs.view(1, -1), labels.view(1, -1))
        loss.backward()
        optimizer.step()


    sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f ' % (i, len(train_df), loss.item()))



    if args.c_path:
        atten_max = atten_max / len(train_df)
        atten_min = atten_min / len(train_df)
        atten_mean = atten_mean / len(train_df)
        with open(log_path, 'a+') as log_txt:
            log_txt.write('\n atten_max' + str(atten_max))
            log_txt.write('\n atten_min' + str(atten_min))
            log_txt.write('\n atten_mean' + str(atten_mean))
    return total_loss / len(train_df)


def test(test_df, milnet, criterion, optimizer, args, log_path, epoch):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    milnet.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for i , (features, labels) in enumerate(test_df):
            outputs = milnet(features)
            preds = torch.argmax(outputs, dim=1)
            # correct += (preds == labels).sum().item()
            # total += labels.size(0)
            loss = criterion(outputs, labels)

        sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
        test_labels.extend(labels)
        if args.average:  # notice args.average here
            test_predictions.extend([(0.5 * torch.sigmoid(outputs) + 0.5 * torch.sigmoid(
                outputs)).squeeze().cpu().numpy()])

        else:
            test_predictions.extend([(0.0 * torch.sigmoid(outputs) + 1.0 * torch.sigmoid(
                outputs)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    test_predictions = np.squeeze(test_predictions, axis=0)

    # nan_mask = ~np.isnan(test_predictions).any(axis=1)

    # 只保留不包含 NaN 的行
    # test_predictions = test_predictions[nan_mask]
    # test_labels = test_labels[nan_mask]

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    with open(log_path, 'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print(confusion_matrix(test_labels, test_predictions))
        info = confusion_matrix(test_labels, test_predictions)
        with open(log_path, 'a+') as log_txt:
            log_txt.write('\n' + str(info))

    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
            print(confusion_matrix(test_labels[:, i], test_predictions[:, i]))
            info = confusion_matrix(test_labels[:, i], test_predictions[:, i])
            with open(log_path, 'a+') as log_txt:
                log_txt.write('\n' + str(info))
    bag_score = 0
    # average acc of all labels
    for i in range(0, len(test_predictions)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_predictions)  # ACC
    cls_report = classification_report(test_labels, test_predictions, digits=4)

    # print(confusion_matrix(test_labels,test_predictions))
    print('\n multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100, sum(auc_value) / len(auc_value) * 100))
    print('\n', cls_report)
    with open(log_path, 'a+') as log_txt:
        log_txt.write('\n  multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100,
                                                                           sum(auc_value) / len(auc_value) * 100))
        log_txt.write('\n' + cls_report)

    return total_loss / len(test_predictions), avg_score, auc_value, thresholds_optimal



def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        if sum(label) == 0:
            continue
        prediction = predictions[:, c]
        # print(label, prediction,label.shape, prediction.shape, labels.shape, predictions.shape)
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def main():
    parser = argparse.ArgumentParser(description='Train our model')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=60, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='xiangya2', type=str, help='Dataset folder name')
    # parser.add_argument('--datasets', default='xiangya2', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='MLP', type=str, help='model our',choices={'MLP', 'FTTransformer'})
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True,
                        help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--agg', type=str, help='which agg')
    parser.add_argument('--c_path', nargs='+',
                        default=None, type=str,
                        help='directory to confounders') #'./datasets_deconf/STAS/train_bag_cls_agnostic_feats_proto_8_transmil.npy'
    # parser.add_argument('--dir', type=str,help='directory to save logs')

    args = parser.parse_args()
    # assert args.model == 'transmil'

    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    if args.c_path:
        save_path = os.path.join('deconf', datetime.date.today().strftime("%m%d%Y"),
                                 str(args.dataset) + '_' + str(args.model) + '_' + str(args.agg) + '_c_path')
    else:
        save_path = os.path.join('baseline_xiangya2', datetime.date.today().strftime("%m%d%Y"),
                                 str(args.dataset) + '_' + str(args.model) + '_' + str(args.agg) + '_fulltune')

    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file, 'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'

# ✅ 简单 MLP 模型


    batch_size = 32

    # ✅ 读取完整数据
    csv_file = 'T:/STAS_multis/data/features.csv'
    label_col = 'label'  # 你需要有一个名为 label 的列
    data = pd.read_csv(csv_file)
    X = data.drop(columns=['filename', label_col])
    y = data[label_col].values

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    fold = 0
    for train_index, val_index in rskf.split(X, y):
        train_path = data.iloc[train_index]
        test_path = data.iloc[val_index]

        # 保存 train/val 到 CSV 文件
        train_file = f'fold{fold}_train.csv'
        val_file = f'fold{fold}_val.csv'
        train_path.to_csv(os.path.join(save_path, train_file), index=False)
        test_path.to_csv(os.path.join(save_path, val_file), index=False)
        print(f"Fold {fold} — train: {len(train_path)}, val: {len(test_path)}")

        # 创建 DataLoader
        train_dataset = FeatureDataset(train_path, label_col=label_col)
        val_dataset = FeatureDataset(test_path, label_col=label_col)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型
        input_dim = train_dataset.features.shape[1]
        num_classes = len(np.unique(y))
        #获取模型
        from Models.CT import MLP, FTTransformer
        if args.model =='MLP':
            model = MLP(input_dim, num_classes)
        elif args.model == 'FTTransformer':
            model = FTTransformer(num_features=input_dim, num_classes=num_classes)


        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        best_score = 0

        for epoch in range(1, args.num_epochs):
            start_time = time.time()
            train_loss_bag = train(train_loader, model, criterion, optimizer, args, log_path)  # iterate all bags
            print('epoch time:{}'.format(time.time() - start_time))
            test_loss_bag, avg_score, aucs, thresholds_optimal = test(val_loader, model, criterion, optimizer, args,
                                                                      log_path, epoch)

            info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % (
                epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join(
                'class-{}>>{}'.format(*k) for k in enumerate(aucs)) + '\n'
            with open(log_path, 'a+') as log_txt:
                log_txt.write(info)
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' %
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join(
                'class-{}>>{}'.format(*k) for k in enumerate(aucs)))
            if args.model != 'transmil':
                scheduler.step()
            current_score = (sum(aucs) + avg_score) / 2
            if current_score >= best_score:
                best_score = current_score
                save_name = os.path.join(save_path, str(run + 1) + f'_{fold}.pth')
                torch.save(model.state_dict(), save_name)
                with open(log_path, 'a+') as log_txt:
                    info = 'Best model saved at: ' + save_name + '\n'
                    log_txt.write(info)
                    info = 'Best thresholds ===>>> ' + '|'.join(
                        'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)) + '\n'
                    log_txt.write(info)
                print('Best model saved at: ' + save_name)
                print(
                    'Best thresholds ===>>> ' + '|'.join(
                        'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            if epoch == args.num_epochs - 1:
                save_name = os.path.join(save_path, f'_{fold}_last.pth')
                torch.save(model.state_dict(), save_name)
        log_txt.close()

        fold += 1


if __name__ == '__main__':
    main()