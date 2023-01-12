# -*- coding: utf-8 -*-
# # Source Free Knowledge Transfer for Privacy-Preserving Unsupervised Motor Imagery Classification
# Author: Wen Zhang and Dongrui Wu
# Date: Oct., 2021
# E-mail: wenz@hust.edu.cn

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from data_align import centroid_align


def read_bci_data(idt, data_name='001-2014', align=False, cov_type='lwf'):
    file = './data/' + data_name + '.npz'
    # (9, 288, 22, 750) (9, 288)

    MI = np.load(file)
    Data_raw = MI['data']
    n_sub = len(Data_raw)
    Label = MI['label']

    # MTS transfer
    tar_data = np.squeeze(Data_raw[idt, :, :, :])
    tar_label = np.squeeze(Label[idt, :])
    if align:
        ca = centroid_align(center_type='riemann', cov_type=cov_type)  # 'lwf', 'oas'
        _, tar_data = ca.fit_transform(tar_data)

    covar_src = Covariances(estimator=cov_type).transform(tar_data)
    tar_fea = TangentSpace().fit_transform(covar_src)

    # MTS transfer
    ids = np.delete(np.arange(0, n_sub), idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        tmp_data = np.squeeze(Data_raw[ids[i]])
        tmp_lbl = np.squeeze(Label[ids[i]])

        if align:
            ca = centroid_align(center_type='riemann', cov_type=cov_type)  # 'lwf', 'oas'
            _, tmp_data = ca.fit_transform(tmp_data)

        src_data.append(tmp_data)
        src_label.append(tmp_lbl)
    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)
    src_label = np.squeeze(src_label)
    covar_tar = Covariances(estimator=cov_type).transform(src_data)
    src_fea = TangentSpace().fit_transform(covar_tar)

    # X.shape - (#samples, # feas)
    # print(src_fea.shape, src_label.shape)

    return src_fea, src_label, tar_fea, tar_label

