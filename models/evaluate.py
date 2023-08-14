import os
import sys
from os.path import dirname, abspath

project_path = dirname(dirname(abspath(__file__)))
model2_path = os.path.join(project_path, 'model2')
sys.path.append(project_path)
sys.path.append(model2_path)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
import pickle
import torch
import torch.nn.functional as F


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from model2 import model_by_type, fc_in_features, num_workers
from load import load_data, load_model, load_history, settings_data, folder_dir_config, model_path_extractor
from config import preprocess_dir, verbose


# models_names = ['multi_all_mode1', 'multi_all_mode2', 'multi_all_mode3', 'multi_all_mode4']
models_names = ['multi_all_mode3']
# Examples type:
# Seperated:  X10_0 (bottom), X10_1 (top), X10_both
examples_by_views = ['V1', 'V1', 'V2', 'V2', 'V3', 'V3', 'V4', 'V4', 'V5', 'V5']
examples_types = [['all', 'all'], ['all', 'all'], ['all', 'all'], ['all', 'all']]  # ['X10_OV', 'X10_OV'], ['X20_OV', 'X20_OV']
examples_types = [examples_by_views, ['X10_OV', 'X10_OV', 'X20_OV', 'X20_OV'], examples_by_views]
examples_types = [['all']]

explain = False
# cur_date = '30_01_2023'  # date of the chosen model
cur_date = '17_01_2023'  # date of the chosen model
data_path = preprocess_dir  # directory of the data set after pre-process
eval_all_data = False
folder_dir = folder_dir_config(fc_in_features, cur_date, explain)


def predict(inputs, labels, model):
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    # df = pd.DataFrame({'Group Name': global_group_names, 'Model Prediction': preds.cpu(), 'Labels': labels.cpu()})
    # df.to_csv(f'{save_dir}/LIME_output_with_yellow.csv')
    preds = np.array(preds.cpu())
    labels = np.array(labels.cpu())
    outputs = np.array(F.softmax(outputs, dim=1).cpu())
    if verbose > 2:
        print(f'outputs: {outputs} \n'
              f'preds: {preds} \n '
              f'labels: {labels} \n')
    return outputs, preds, labels


def save(save_dict):
    for key, value in save_dict.items():
        with open(os.path.join(save_dir, f'{key}_{value["info"]}.pkl'), 'wb') as f:
            pickle.dump(value['file'], f)


def plot_fpr_tpr(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig(os.path.join(save_dir, f'roc_model_by_{model_by_type}.png'))
    plt.close()


def plot_history():
    train_acc_history, train_loss_history, val_acc_history, val_loss_history = load_history(save_dir)
    modes = {'loss': {'val': val_loss_history, 'train': train_loss_history},
             'accuracy': {'val': val_acc_history, 'train': train_acc_history}}
    for mode, info in modes.items():
        for phase, hist in info.items():
            plt.plot(hist, label=f'{phase}_{mode}')
        plt.xlabel('epochs')
        plt.ylabel(mode)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(save_dir, f'{mode}_history.png'), bbox_inches='tight')
        plt.close()


def conf_mat_metrics(outputs, preds, labels):
    fpr, tpr, thresholds = roc_curve(labels, outputs[:, 1])
    auc = roc_auc_score(labels, outputs[:, 1])
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    conf_mat = [tn, fp, fn, tp]
    if verbose > 0:
        print('AUC: %.3f \n' % auc)
    save_dict = {'conf_mat': {'info': f'{tn}_{fp}_{fn}_{tp}_{model_by_type}', 'file': conf_mat},
                 'auc': {'info': f'{int(auc * 100)}_by_{model_by_type}', 'file': auc},
                 'fpr': {'info': f'model_by_{model_by_type}', 'file': fpr},
                 'tpr': {'info': f'model_by_{model_by_type}', 'file': tpr}}
    save(save_dict)
    return conf_mat, fpr, tpr, auc


def auc_calc(data_loader, model, plot):
    torch.manual_seed(1)
    with torch.no_grad():
        for inputs, labels, global_group_names in data_loader:
            conf_mat, fpr, tpr, auc = conf_mat_metrics(*predict(inputs, labels, model))
    if plot:
        plot_fpr_tpr(fpr=fpr, tpr=tpr)


for mode_idx, model_name in enumerate(models_names):
    pretrained_model_path = ''
    pretrained_model_dir = os.path.join(model2_path, f'models/fc_in_features_128_17_01_2023/{model_name}')
    folder_model_dir = os.path.join(folder_dir, f'{model_name}')
    for j, examples_type in enumerate(examples_types[mode_idx]):
        multiview, no_yellow = settings_data(examples_type=examples_type, j=j)
        save_dir = os.path.join(folder_model_dir, f'{examples_type}_{int(no_yellow)}')
        model_type_dir = os.path.join(save_dir, model_name)
        if explain:
            pretrained_model_path = model_path_extractor(os.path.join(pretrained_model_dir, f'all_{int(no_yellow)}'), model_by_type)
        model_path = model_path_extractor(save_dir, model_by_type)
        model_dir = dirname(dirname(save_dir))
        data_loader = load_data(data_path=data_path,
                                model_dir=model_dir,
                                examples_type=examples_type,
                                multiview=multiview,
                                no_yellow=no_yellow,
                                eval=True,
                                all_data=eval_all_data)

        model, device = load_model(model_path=model_path,
                                   pretrained_model_path=pretrained_model_path,
                                   examples_type=examples_type,
                                   fc_in_features=fc_in_features,
                                   mode=int(model_name[-1]),
                                   explain=explain)
        auc_calc(data_loader=data_loader, model=model, plot=True)
        plot_history()
