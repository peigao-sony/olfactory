#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix


# In[ ]:


def szymanski_ts_eq_fold(n_splits, y):

    y_train = lil_matrix(y)

    n_samples = y_train.shape[0] 
    n_labels = y_train.shape[1] 

    percentage_per_fold = [1/float(n_splits) for i in range(n_splits)]
    desired_samples_per_fold = np.array([percentage_per_fold[i]*n_samples for i in range(n_splits)]) 

    folds = [[] for i in range(n_splits)] #10 lists

    samples_with_label = [[] for i in range(n_labels)]

    for sample, labels in enumerate(y_train.rows):
        for label in labels:
            samples_with_label[label].append(sample)
    # labelpair based sample size
            
    samples_with_labelpairs = {}
    for row, labels in enumerate(y_train.rows):
        pairs = [(a, b) for b in labels for a in labels if a <= b]
        for p in pairs:
            if p not in samples_with_labelpairs:
                samples_with_labelpairs[p] = []
            samples_with_labelpairs[p].append(row)

    desired_samples_per_labelpair_per_fold = {k : [len(v)*i for i in percentage_per_fold] for k,v in samples_with_labelpairs.items()}

    labels_of_edges = samples_with_labelpairs.keys() # 20 pairs
    labeled_samples_available = [len(samples_with_labelpairs[v]) for v in labels_of_edges] #XXXXXX
    # labelpair based sample size
    
    rows_used = {i : False for i in range(n_samples)}
    total_labeled_samples_available = sum(labeled_samples_available) #1723
    old_l=None

    while total_labeled_samples_available > 0:
        l = list(labels_of_edges)[np.argmin(np.ma.masked_equal(labeled_samples_available, 0, copy=False))]

        while len(samples_with_labelpairs[l])>0:

            row = samples_with_labelpairs[l].pop()
            if rows_used[row]:
                continue

            max_val = max(desired_samples_per_labelpair_per_fold[l])
            M = np.where(np.array(desired_samples_per_labelpair_per_fold[l])==max_val)[0]
            #print(l, M, len(M))

            m = None
            if len(M) == 1:
                m = M[0]
            else:
                max_val = max(desired_samples_per_fold[M])
                M_bis = np.where(np.array(desired_samples_per_fold)==max_val)[0]
                M_bis = np.array([x for x in M_bis if x in M])
                m = np.random.choice(M_bis, 1)[0]

            folds[m].append(row)
            rows_used[row]=True #----
            desired_samples_per_labelpair_per_fold[l][m]-=1
            if desired_samples_per_labelpair_per_fold[l][m] <0:
                desired_samples_per_labelpair_per_fold[l][m]=0

            for i in samples_with_labelpairs.keys():
                if row in samples_with_labelpairs[i]:
                    samples_with_labelpairs[i].remove(row)
                    desired_samples_per_labelpair_per_fold[i][m]-=1

                if desired_samples_per_labelpair_per_fold[i][m] <0:
                    desired_samples_per_labelpair_per_fold[i][m]=0
            desired_samples_per_fold[m]-=1

        labeled_samples_available = [len(samples_with_labelpairs[v]) for v in labels_of_edges]
        total_labeled_samples_available = sum(labeled_samples_available)

        available_samples = [i for i, v in rows_used.items() if not v]
        samples_left = len(available_samples)


    assert (samples_left + sum(map(len, folds))) == n_samples

    while samples_left>0:
        row = available_samples.pop()
        rows_used[row]=True
        fold_selected = np.random.choice(np.where(desired_samples_per_fold>0)[0], 1)[0]
        folds[fold_selected].append(row)
        samples_left-=1

    assert sum(map(len, folds)) == n_samples
    assert len([i for i, v in rows_used.items() if not v])==0
    return folds

