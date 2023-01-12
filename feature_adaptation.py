# -*- coding: utf-8 -*-
# # Lightweight Source-Free Transfer for Privacy-Preserving Motor Imagery Classification
# Author: Wen Zhang and Dongrui Wu
# E-mail: wenz@hust.edu.cn
# Refer: https://github.com/chamwen/JPDA

import numpy as np
import scipy.io
import scipy.linalg
from sklearn import preprocessing
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.preprocessing import OneHotEncoder
import argparse


class feature_adaptation:
    def __init__(self, args):
        self.kernel_type = args.kernel_type
        self.mmd_type = args.mmd_type
        self.dim = args.dim
        self.lamb = args.lamb
        self.gamma = args.gamma
        self.mu = args.mu
        self.T = args.T

    def fit_predict(self, Xs, Ys, Xt, Y_tar_pseudo):
        X = np.hstack((Xs.T, Xt.T))
        X = np.dot(X, np.diag(1. / np.linalg.norm(X, axis=0)))
        m, n = X.shape  # 800, 2081
        ns, nt = len(Xs), len(Xt)

        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = get_matrix_M(Ys, Y_tar_pseudo, ns, nt, C, self.mu, mmd_type=self.mmd_type)

        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        return Z


def get_matrix_M(Ys, Y_tar_pseudo, ns, nt, C, mu, mmd_type='djp-mmd'):
    M = 0
    if mmd_type == 'jmmd':
        N = 0
        n = ns + nt
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M0 = e * e.T * C
        if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
            for c in np.unique(Ys):
                e = np.zeros((n, 1))
                tt = Ys == c
                e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                yy = Y_tar_pseudo == c
                ind = np.where(yy == True)
                inds = [item + ns for item in ind]
                e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                e[np.isinf(e)] = 0
                N = N + np.dot(e, e.T)
        M = M0 + N
        M = M / np.linalg.norm(M, 'fro')

    if mmd_type == 'djp-mmd':
        ohe = OneHotEncoder()
        ohe.fit(np.unique(Ys).reshape(-1, 1))
        Ys_ohe = ohe.transform(Ys.reshape(-1, 1)).toarray().astype(np.int8)

        # For transferability
        Ns = 1 / ns * Ys_ohe
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Yt_ohe = ohe.transform(Y_tar_pseudo.reshape(-1, 1)).toarray().astype(np.int8)
            Nt = 1 / nt * Yt_ohe
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        Rmin = Rmin / np.linalg.norm(Rmin, 'fro')

        # # For discriminability
        Ms = np.zeros([ns, (C - 1) * C])
        Mt = np.zeros([nt, (C - 1) * C])
        for i in range(C):
            idx = np.arange((C - 1) * i, (C - 1) * (i + 1))
            Ms[:, idx] = np.tile(Ns[:, i], (C - 1, 1)).T
            tmp = np.arange(C)
            Mt[:, idx] = Nt[:, tmp[tmp != i]]
        Rmax = np.r_[np.c_[np.dot(Ms, Ms.T), np.dot(-Ms, Mt.T)], np.c_[np.dot(-Mt, Ms.T), np.dot(Mt, Mt.T)]]
        Rmax = Rmax / np.linalg.norm(Rmax, 'fro')
        M = Rmin - mu * Rmax

    return M


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


if __name__ == '__main__':
    domains = ['caltech_SURF_L10.mat', 'amazon_SURF_L10.mat', 'webcam_SURF_L10.mat', 'dslr_SURF_L10.mat']
    name_list = [name[0].upper() for name in domains]
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

    num_domain = len(domains)
    acc_all = np.zeros([len(name_list) * (len(name_list) - 1)])
    itr_idx = 0
    for s in range(num_domain):  # source
        for t in range(num_domain):  # target
            if s != t:
                print('%s: %s --> %s' % (itr_idx, name_list[s], name_list[t]))
                src, tar = 'data/' + domains[s], 'data/' + domains[t]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['fts'], src_domain['labels'], tar_domain['fts'], tar_domain['labels']

                # can only be added in offline learning, follow JDA original code
                Xs = preprocessing.scale(Xs)
                Xt = preprocessing.scale(Xt)

                # # I: joint MMD
                mmd_list = ['jmmd', 'djp-mmd']
                mmd_type = mmd_list[0]
                args = argparse.Namespace(kernel_type='primal', mmd_type=mmd_type,
                                          dim=30, lamb=1, gamma=1, mu=0.1, T=5)

                Y_tar_pseudo_hard = None
                list_acc = []
                for itr in range(args.T):
                    Z = feature_adaptation(args).fit_predict(Xs, Ys, Xt, Y_tar_pseudo_hard)
                    Xs_new, Xt_new = Z[:, :len(Xs)].T, Z[:, len(Xs):].T

                    clf = SVC()
                    # clf = KNeighborsClassifier(n_neighbors=1)
                    clf.fit(Xs_new, Ys.ravel())
                    Y_tar_pseudo = clf.predict(Xt_new)
                    Y_tar_pseudo_hard = Y_tar_pseudo
                    acc = accuracy_score(Yt, Y_tar_pseudo_hard)
                    list_acc.append(acc)
                    print('iteration [{}/{}]: acc: {:.4f}'.format(itr + 1, args.T, acc))
                acc_all[itr_idx] = list_acc[-1]
                print('type: {} -- acc: {:.4f}\n'.format(mmd_type, list_acc[-1]))
                itr_idx += 1

    print('mean acc...')
    print(np.round(np.mean(acc_all, axis=0), 4))
    print(np.round(acc_all, 4))
