# -*- coding: utf-8 -*-
# # Lightweight Source-Free Transfer for Privacy-Preserving Motor Imagery Classification
# Author: Wen Zhang and Dongrui Wu
# E-mail: wenz@hust.edu.cn

import numpy as np
import joblib
import argparse
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from feature_adaptation import feature_adaptation
from dataloader import read_bci_data

import warnings

warnings.filterwarnings("ignore")


def pre_train_models_combine_voting(Xs, Ys, root_path, mdl_list):
    clf1 = SVC(probability=True)
    clf2 = LDA(solver='lsqr', shrinkage='auto')
    clf3 = LogisticRegression(penalty='l2', max_iter=500)
    clf_dict = dict(svc_soft=clf1, lda_soft=clf2, lr_soft=clf3)
    for idx in range(len(mdl_list)):
        clf_base = clf_dict[mdl_list[idx]]
        clf_base.fit(Xs, Ys.ravel())
        mdl_path = root_path + 'mdl_' + str(idx) + '.pkl'
        joblib.dump(clf_base, mdl_path)


def get_virtual_mid_source(Xt, Y_tar_pseudo_list, std_th):
    Y_tar_vote = np.squeeze(np.mean(Y_tar_pseudo_list, axis=0))

    ins_std_all = []
    for ni in range(len(Xt)):
        tmp_prob = np.squeeze(Y_tar_pseudo_list[:, ni, :])
        tmp_std = np.mean([np.std(tmp_prob[:, i]) for i in range(tmp_prob.shape[1])])
        ins_std_all.append(tmp_std)

    idx_select = np.where(np.array(ins_std_all) < std_th)[0]  # std_th 0.1
    print('select ins num ', len(idx_select))
    Ys_mid = Y_tar_vote[idx_select, :].argmax(axis=1)
    Xs_mid = Xt[idx_select, :]
    print(Xs_mid.shape, Ys_mid.shape)
    return Xs_mid, Ys_mid


data_name_list = ['001-2014_2', '001-2014', '001-2015', 'AlexMI']
data_name = data_name_list[0]
if data_name == '001-2014_2': num_sub, chn = 9, 22  # MI2-2
if data_name == '001-2014': num_sub, chn = 9, 22  # MI2-4
if data_name == '001-2015': num_sub, chn = 12, 13  # MI2015
if data_name == 'AlexMI': num_sub, chn = 8, 16  # AlexMI

# Generate Source Models except target sub t
mdl_list = ['svc_soft', 'lda_soft', 'lr_soft']
# for t in range(num_sub):
#     print('target', t)
#     Xs, Ys, Xt, Yt = read_bci_data(t, data_name, align=True, cov_type='lwf')
#     root_path = './models_voting/' + data_name + '_TL_src_' + str(t) + '_'
#     pre_train_models_combine_voting(Xs, Ys, root_path, mdl_list)
# print('finished pre-training...\n')

# cross subject classification
acc_tl_all = np.zeros(num_sub, )
for tar in range(num_sub):
    print('\n====================== Transfer to S' + str(tar + 1) + ' ======================')
    Xs, Ys, Xt, Yt = read_bci_data(tar, data_name, align=True, cov_type='lwf')

    ###############################################################################
    # TSM+LSFT
    root_path = './models_voting/' + data_name + '_TL_src_' + str(tar) + '_'
    Y_tar_pseudo_list = []
    mdl_idx = [1, 2]
    for idx in mdl_idx:
        clf = joblib.load(root_path + 'mdl_' + str(idx) + '.pkl')
        Y_tar_pseudo = clf.predict_proba(Xt)
        Y_tar_pseudo_list.append(Y_tar_pseudo)
    Y_tar_pseudo_list = np.array(Y_tar_pseudo_list)
    Y_tar_vote = np.squeeze(np.mean(Y_tar_pseudo_list, axis=0))
    print(np.array(mdl_list)[mdl_idx])

    # Construct Virtual Intermediate Source Domain
    std_th = 0.1
    Xs_mid, Ys_mid = get_virtual_mid_source(Xt, Y_tar_pseudo_list, std_th)
    ns_mid = len(Ys_mid)

    # Feature Adaptation
    args = argparse.Namespace(kernel_type='primal', mmd_type='djp-mmd',
                              dim=20, lamb=1, gamma=1, mu=0.1, T=10)

    Y_tar_pseudo = Y_tar_vote.argmax(axis=1) + np.unique(Ys)[0]
    list_acc = []
    for itr in range(args.T):
        Z = feature_adaptation(args).fit_predict(Xs_mid, Ys_mid, Xt, Y_tar_pseudo)
        Xs_new, Xt_new = Z[:, :ns_mid].T, Z[:, ns_mid:].T
        clf = LDA(solver='lsqr', shrinkage='auto')
        clf.fit(Xs_new, Ys_mid.ravel())
        Y_tar_pseudo = clf.predict(Xt_new)

        acc = accuracy_score(Yt, Y_tar_pseudo)
        list_acc.append(acc)
        print('iteration [{}/{}]: acc: {:.4f}'.format(itr + 1, args.T, acc))
    acc_tl_all[tar] = list_acc[-1]
    print('** LSFT: tar S%d' % (tar + 1), ' -- acc %.4f' % acc_tl_all[tar])

print(np.array(mdl_list)[mdl_idx])
print('\nCA+LSFT acc mean | std')
print(np.round(np.mean(acc_tl_all) * 100, 2), np.round(np.std(acc_tl_all) * 100, 2))

###############################################################################
# CA+LSFT (lda+lr)
# 75.15  MI2-2
# 57.99  MI2-4
# 62.33  MI2015
# 76.88  AlexMI
