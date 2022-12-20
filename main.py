#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import numpy as np
import pandas as pd
import random
import time

import argparse

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx

import shutil
import os
from os import listdir
from os.path import isfile, join

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.variable import Variable
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import pickle
import copy
import datetime
import jsonpickle
from itertools import chain


from functions.SecondOrder import szymanski_ts_eq_fold
from functions.GraphReader import qm9_edges,qm9_nodes
from functions.GraphReader import get_values,get_graph_stats

from functions.Utlis import AverageMeter,collate_g

from models.MPNN import MPNN




dat_label = pd.read_csv('dataset.csv',index_col=0)
dat_smiles = pd.read_csv('smiles.csv', index_col=0)
dict_smile = dat_smiles.to_dict()['SMILES']

dat_labels = dat_label.T


from sklearn.utils.class_weight import compute_class_weight
lst_imbalance_ratio = []
for i in dat_label.values:
    class_weights=compute_class_weight('balanced',classes=np.unique(i), y=i)
    ratio = class_weights[1]
    lst_imbalance_ratio.append(ratio)

import math
weights = []
for i in range(len(lst_imbalance_ratio)):
    weights.append(math.log(1+lst_imbalance_ratio[i]))
weights = torch.tensor(weights)



from scipy.sparse import lil_matrix
lil_matrix(dat_labels.values)

n_splits = 5

folds =szymanski_ts_eq_fold(n_splits, dat_labels)

print('n-folds splitting completed')

# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

parser = argparse.ArgumentParser(description='Neural message passing')

#parser.add_argument('--dataset', default='qm9', help='QM9')
#parser.add_argument('--datasetPath', default='/home/taobai/Documents/Model/MPNN/mpnn-data/qm9/dsgdb9nsd/', help='dataset path')
#parser.add_argument('--logPath', default='./log/qm9/mpnn/', help='log path')
#parser.add_argument('--plotLr', default=False, help='allow plotting the data')
#parser.add_argument('--plotPath', default='./plot/qm9/mpnn/', help='plot path')
#parser.add_argument('--resume', default='./checkpoint/qm9/mpnn/',
#                    help='path to latest checkpoint')

# Optimization Options
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
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
# parser.add_argument('--model', type=str,help='MPNN model name [MPNN, MPNNv2, MPNNv3]',default='MPNN')


args = parser.parse_args()
print(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()



best_er1 = 0



def xyz_graph_reader(CID):
    dict_smile = np.load('dict_smile.npy',allow_pickle=True).item()
    smiles = dict_smile[CID]
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
   
    g = nx.Graph()
    # Create nodes
    for i in range(0, m.GetNumAtoms()):
        atom_i = m.GetAtomWithIdx(i)

        g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                   aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                   num_h=atom_i.GetTotalNumHs())


    # Read Edges
    for i in range(0, m.GetNumAtoms()):
        for j in range(0, m.GetNumAtoms()):
            e_ij = m.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j, b_type=e_ij.GetBondType())
            else:
                # Unbonded
                g.add_edge(i, j, b_type=None)

    l = list(dat_label[CID].values)           
                
    return g , l


# In[12]:


class Qm9():
#class Qm9(data.Dataset):
    # Constructor
    def __init__(self, idx, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                 target_transform=None, e_representation='raw_distance'):
        self.idx = idx
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.e_representation = e_representation

    def __getitem__(self,index):
        
        g, target = xyz_graph_reader(self.idx[index])
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)

        if self.edge_transform is not None:
            g, e = self.edge_transform(g)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #g：adjacent matrix
        #h：node properties（list of list）
        #e：diction，key:edge，value:properties
        return (g, h, e), target

    def __len__(self):
        return len(self.idx)

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform



result_cl = []
result_auc = []
initial_lr = args.lr

evaluation = lambda output, target: torch.eq(torch.round(output),target).float().mean()
# In[ ]:


def train(train_loader, model, criterion, optimizer, epoch, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    
    #list_loss = []

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
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, err=accuracy))

    print('Epoch: [{0}] Accuracy {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, err=accuracy, loss=losses, b_time=batch_time))


def validate(val_loader, model, criterion, evaluation, logger=None):
#     batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (g, h, e, target) in enumerate(val_loader):

        # Prepare input data
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Compute output
        output = model(g, h, e)

        # Logs
        losses.update(criterion(output, target).item(), g.size(0))
        accuracy.update(evaluation(output, target).item(), g.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

    print(' * Average accuracy {err.avg:.3f}; '
          .format(err=accuracy))


    return accuracy.avg



def main(args):
    for n_folds in range(5):
        args.lr = initial_lr
        idxes = []
        for i in range(5):
            idxes.append(i)

        test_folds = int(n_folds)
        train_folds = idxes[:]
        train_folds.remove(n_folds)

        test_ids = [dat_label.columns[i] for i in folds[test_folds]]
        train_ids =  [dat_label.columns[i] for j in train_folds for i in folds[j]]

        data_train = Qm9(train_ids, edge_transform=qm9_edges, e_representation='raw_distance')
        data_test = Qm9(test_ids, edge_transform=qm9_edges, e_representation='raw_distance')

        # Select one graph
        g_tuple, l = data_train[0]
        g, h_t, e = g_tuple

        stat_dict = get_graph_stats(data_train[0])

        # set_target_transform was defined in Class Qm9
        data_train.set_target_transform(lambda x:x)
        data_test.set_target_transform(lambda x: x)


        # collate_g from utils
        train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,shuffle=True, collate_fn=collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=collate_g,
                                              num_workers=args.prefetch, pin_memory=True)

        # feature dimension of nodes and edges
        in_n = [len(h_t[0]), len(list(e.values())[0])] 
        # dimension of hidden state/embedding
        hidden_state_size = 20
        # dimension of message 
        message_size = 20
        # layers of GNN
        n_layers = 3
        # num of labels
        l_target = len(l)
        type ='classification'

        model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type=type)

        del in_n, hidden_state_size, message_size, n_layers, l_target, type

        #print('Optimizer')
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
        for epoch in range(0, args.epochs):
            torch.cuda.empty_cache()

            if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
                args.lr -= lr_step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on test set
            #er1 = validate(valid_loader, model, criterion, evaluation)

            #is_best = er1 > best_er1
            #best_er1 = min(er1, best_er1)

        # For testing
        report = []
        all_pred = []
        all_target =[]
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()

        if args.cuda:
            print('\t* Cuda')
            model = model.cuda()
            criterion = criterion.cuda()
        for i, (g, h, e, target) in enumerate(test_loader):
            torch.cuda.empty_cache()
            # Prepare input data
            if args.cuda:
                g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
            g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

            # Compute output
            output =model(g, h, e)
            preds = torch.round(output)

            all_pred.append(preds.cpu().detach().numpy())
            all_target.append(target.cpu().detach().numpy())

        all_pred = np.vstack(all_pred)
        all_target = np.vstack(all_target)

        lst_auc = []
        for i in range(len(l)):
            lst_auc.append(roc_auc_score(np.transpose(all_target)[i], np.transpose(all_pred)[i]))
        lst_auc.append(roc_auc_score(all_target, all_pred))

        cl = classification_report(all_target, all_pred,output_dict=True)
        report = pd.DataFrame.from_dict(cl, orient='index')

        result_cl.append(report)
        result_auc.append(lst_auc)

        del model,optimizer,g, h, e, target, output, preds, criterion
        torch.cuda.empty_cache()

    results_cl = result_cl[0]
    for i in range(1,5):
        results_cl = results_cl + result_cl[i]
    results_auc = (np.sum([i for i in result_auc], axis = 0))
    
    results_cl.drop(index=['micro avg','weighted avg','samples avg'], inplace=True)
    results_cl.insert(results_cl.shape[1]-1, 'AUC', results_auc/5) 
    results_cl = pd.concat([results_cl.iloc[:,:3]/5,results_cl.iloc[:,3:]],axis=1)
    results_cl.to_csv('cl_report.csv')

if __name__ == '__main__':
    main(args)

