#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import numpy as np
import pandas as pd
import random
import time
import argparse
from scipy.sparse import lil_matrix

from rdkit import Chem
import networkx as nx

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import math

from functions.SecondOrder import szymanski_ts_eq_fold
from functions.GraphReader import qm9_edges,qm9_nodes,xyz_graph_reader,Qm9
from functions.GraphReader import get_values,get_graph_stats
from functions.Utlis import AverageMeter,collate_g
from functions.Mydatasets import Mydatasets
from models.MPNN import MPNN
from models.MLP import MLP


# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

# Optimization Options
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='Number of epochs to train (default: 100)')
parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-4, metavar='LR',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.5, metavar='LR-DECAY',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')
parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                    help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# i/o
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='How many batches to wait before logging training status')
# Accelerating
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

# Model modification
parser.add_argument('--model', type=str,help='model name [MPNN, MLP]',default='MLP')


args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
#args =parser.parse_known_args()[0]


dat_label = pd.read_csv('dat_CID_labels.csv',index_col=0) # Load the label
if args.model == 'MLP':
    dat_mordred = pd.read_csv('dat_CID_mordred.csv',index_col=0)
if args.model == 'MPNN':
    dat_smiles = pd.read_csv('dat_CID_SMILES.csv', index_col=0)
    #dict_smile = dat_smiles.to_dict()['SMILES']

lst_imbalance_ratio = []
for i in dat_label.values:
    class_weights=compute_class_weight('balanced',classes=np.unique(i), y=i)
    ratio = class_weights[1]
    lst_imbalance_ratio.append(ratio)
weights = []
for i in range(len(lst_imbalance_ratio)):
    weights.append(math.log(1+lst_imbalance_ratio[i]))
weights = torch.tensor(weights) # calculate the weights of labels and pass to weighted CE

lil_matrix(dat_label.T.values)
n_splits = 5
folds =szymanski_ts_eq_fold(n_splits, dat_label.T)
print('n-folds splitting completed')

result_cl = []
result_auc = []
initial_lr = args.lr

evaluation = lambda output, target: torch.eq(torch.round(output),target).float().mean()

def main(args):
    for n_folds in range(5):
        print('Folds:  {}  '.format(n_folds + 1))
        args.lr = initial_lr
        
        idxes = []
        for i in range(5):
            idxes.append(i)

        test_folds = int(n_folds)
        train_folds = idxes[:]
        train_folds.remove(n_folds)

        test_ids = [dat_label.columns[i] for i in folds[test_folds]]
        train_ids =  [dat_label.columns[i] for j in train_folds for i in folds[j]]
        
        if args.model == 'MPNN':
            data_train = Qm9(train_ids, edge_transform=qm9_edges, e_representation='raw_distance')
            data_test = Qm9(test_ids, edge_transform=qm9_edges, e_representation='raw_distance')

            g_tuple, l = data_train[0]
            g, h_t, e = g_tuple
            stat_dict = get_graph_stats(data_train[0])

            data_train.set_target_transform(lambda x:x)
            data_test.set_target_transform(lambda x: x)

            train_loader = torch.utils.data.DataLoader(data_train,
                                                   batch_size=args.batch_size,shuffle=True, collate_fn=collate_g,
                                                   num_workers=args.prefetch, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(data_test,
                                                  batch_size=args.batch_size, collate_fn=collate_g,
                                                  num_workers=args.prefetch, pin_memory=True)

            in_n = [len(h_t[0]), len(list(e.values())[0])] # feature dimension of nodes and edges
            hidden_state_size = 20 # dimension of hidden state/embedding
            message_size = 20 # dimension of message 
            n_layers = 3 # layers of GNN
            l_target = len(l) # num of labels
            type ='classification'

            model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type=type)
            del in_n, hidden_state_size, message_size, n_layers, l_target, type
            
        elif args.model == 'MLP':
            train_label = torch.tensor(dat_label[train_ids].T.to_numpy(), dtype=torch.float)
            test_label = torch.tensor(dat_label[test_ids].T.to_numpy(), dtype=torch.float)
            train_dat = torch.tensor(dat_mordred.T[train_ids].T.to_numpy(), dtype=torch.float)
            test_dat = torch.tensor(dat_mordred.T[test_ids].T.to_numpy(), dtype=torch.float)

            train_data_set = Mydatasets(data1 = train_dat, label = train_label)
            test_data_set = Mydatasets(data1 = test_dat, label = test_label)

            train_loader = torch.utils.data.DataLoader(train_data_set, batch_size = args.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_data_set, batch_size = args.batch_size, shuffle=False)
            
            n_label = dat_label.shape[0]
            n_feature = dat_mordred.shape[1]
            model = MLP(n_feature,n_label)
        else:
            model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type=type)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss(weight=weights,reduction='mean')
        lr_step = (args.lr-args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])

        print('Check cuda')
        if args.cuda:
            print('\t* Cuda')
            model = model.cuda()
            criterion = criterion.cuda()
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        # Epoch for loop
        for epoch in range(args.epochs):
            torch.cuda.empty_cache()

            if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
                args.lr -= lr_step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

            if args.model == 'MPNN':
                train_MPNN(train_loader, model, criterion, optimizer, epoch)
            elif args.model == 'MLP':
                train_MLP(train_loader, model, criterion, optimizer, epoch)
            else:
                train_MLP(train_loader, model, criterion, optimizer, epoch)
                
        # For testing
        report,all_pred,all_target = [],[],[]
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # model.eval()  # switch to evaluate mode
        
        if args.cuda:
            model = model.cuda()
            criterion = criterion.cuda()
            
        if args.model == 'MPNN':
            for i, (g, h, e, target) in enumerate(test_loader):
                torch.cuda.empty_cache()
                if args.cuda:
                    g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
                g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

                output =model(g, h, e)
                preds = torch.round(output)

                all_pred.append(preds.cpu().detach().numpy())
                all_target.append(target.cpu().detach().numpy())
        elif args.model == 'MLP':
            for data,target in test_loader:
                torch.cuda.empty_cache()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                
                output = model(data)
                preds = torch.round(output)
                
                all_pred.append(preds.cpu().detach().numpy())
                all_target.append(target.cpu().detach().numpy())
        else:
            break

        all_pred = np.vstack(all_pred)
        all_target = np.vstack(all_target)

        lst_auc = []
        for i in range(dat_label.shape[0]):
            lst_auc.append(roc_auc_score(np.transpose(all_target)[i], np.transpose(all_pred)[i]))
        lst_auc.append(roc_auc_score(all_target, all_pred))

        cl = classification_report(all_target, all_pred,output_dict=True)
        report = pd.DataFrame.from_dict(cl, orient='index')

        result_cl.append(report)
        result_auc.append(lst_auc)

        torch.cuda.empty_cache()


    results_cl = result_cl[0]
    for i in range(1,5):
        results_cl = results_cl + result_cl[i]
    results_auc = (np.sum([i for i in result_auc], axis = 0))
    
    results_cl.drop(index=['micro avg','weighted avg','samples avg'], inplace=True)
    results_cl.insert(results_cl.shape[1]-1, 'AUC', results_auc/5) 
    results_cl = pd.concat([results_cl.iloc[:,:3]/5,results_cl.iloc[:,3:]],axis=1)
    results_cl.to_csv('cl_report.csv')

def train_MPNN(train_loader, model, criterion, optimizer, epoch, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (g, h, e, target) in enumerate(train_loader):

        # Prepare input data
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # Compute output
        output = model(g, h, e)
        train_loss = criterion(output, target)
        #list_loss.append(train_loss.item())
        # Logs
        losses.update(train_loss.item(), g.size(0))
        accuracy.update(evaluation(output, target).item(), g.size(0))
        
        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {err.val:.4f} ({err.avg:.4f})'
                  .format(epoch, i*args.batch_size, len(train_loader)*args.batch_size, batch_time=batch_time,
                          data_time=data_time, loss=losses, err=accuracy))

    print('Epoch: [{0}] Accuracy {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, err=accuracy, loss=losses, b_time=batch_time))

def train_MLP(train_loader, model, criterion, optimizer, epoch, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i,(data,target) in enumerate(train_loader):

        # Prepare input data
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        output = model(data)
        train_loss = criterion(output, target)
        
        losses.update(train_loss.item(), args.batch_size)
        accuracy.update(evaluation(output, target).item(), args.batch_size)
        
        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {err.val:.4f} ({err.avg:.4f})'
                  .format(epoch, i*args.batch_size, len(train_loader)*args.batch_size, batch_time=batch_time,
                          data_time=data_time, loss=losses, err=accuracy))

    print('Epoch: [{0}] Accuracy {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, err=accuracy, loss=losses, b_time=batch_time))

if __name__ == '__main__':
    main(args)