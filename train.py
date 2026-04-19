import torch
import random
import sys
import copy

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean, scatter_max, scatter_add
from tqdm import tqdm

from utils import EarlyStopping
from tools.rand5fold import loadfolddata, loadNewBiData
from tools.evaluate import evaluation4class
import sklearn.metrics as metrics


# Hyper-parameters
datasetname = "Twitter15"  # "Twitter15", "Twitter16"
foldnum = 0
lr = 0.0001
weight_decay = 5e-5
patience = 10
n_epochs = 10
batchsize = 256
base_dir = 'data/'

model_name = "GCN"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_accs = []
NR_F1 = []
FR_F1 = []
TR_F1 = []
UR_F1 = []


def train(x_train, x_test, lr, weight_decay, patience, n_epochs, batchsize):
    model = TripleGCNNet(16, 256, 64, pooling='scatter_mean').to(device)
    print(model)
    GNN_params = list(map(id, model.gnn.conv1.parameters()))
    GNN_params += list(map(id, model.gnn.conv2.parameters()))
    GNN_params += list(map(id, model.gnn.conv3.parameters()))
    base_params = filter(lambda p: id(p) not in GNN_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.gnn.conv1.parameters(), 'lr': lr / 5},
        {'params': model.gnn.conv2.parameters(), 'lr': lr / 5},
        {'params': model.gnn.conv3.parameters(), 'lr': lr / 5}
    ], lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadNewBiData(x_train, x_test)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            emb, out_labels = model(Batch_data)
            finalloss = F.nll_loss(out_labels, Batch_data.y)
            loss = finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(
                epoch, batch_idx, loss.item(), train_acc))
            batch_idx += 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            optimizer.zero_grad()
            Batch_data.to(device)
            val_emb, val_out = model(Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(
            epoch, np.mean(temp_val_losses), np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', datasetname)
        accs = np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
        torch.cuda.empty_cache()
    return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4


def main():
    train_data, test_data = loadfolddata(datasetname, foldnum, base_dir)
    train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train(
        train_data, test_data, lr, weight_decay, patience, n_epochs, batchsize)


if __name__ == '__main__':
    main()
