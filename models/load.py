import os
import io
import numpy as np

import pdb
import torch
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model2 import MVCNN, ExpMVCNN, get_explain_component
from model2 import ObjectsDataset, models2_dir, multiview_arr
from config import verbose


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def model_path_extractor(path, model_by_type):
    return [os.path.join(path, fname) for fname in os.listdir(path) if fname.startswith(f'model_by_{model_by_type}')][0]


def folder_dir_config(fc_in_features, cur_date, explain):
    folder_dir = os.path.join(models2_dir, f'fc_in_features_{fc_in_features}_{cur_date}')
    if explain:
        folder_dir = f'{folder_dir}_ex'
    if verbose > 0:
        print(folder_dir)
    return folder_dir


def settings_data(examples_type, j):
    no_yellow = True if (j+1) % 2 == 0 else False
    multiview = True if examples_type in multiview_arr else False
    if verbose > 0:
        print('evaluating...')
        print(f'Examples type: {examples_type}')
        print(f'Multiview: {multiview}')
        print(f'No yellow: {no_yellow}\n')
    return multiview, no_yellow


def load_history(save_dir):
    fnames = ['train_acc_history.pkl', 'train_loss_history.pkl', 'val_acc_history.pkl', 'val_loss_history.pkl']
    history = []
    for fname in fnames:
        with open(os.path.join(save_dir, fname), 'rb') as f:
            history.append(CPU_Unpickler(f).load())
    return history


def load_data(data_path, model_dir, examples_type, multiview, no_yellow, eval=False, sampler_mode='val', verbose=1):
    dataset = ObjectsDataset(data_path=data_path,
                             multiview=multiview,
                             augmentation=False,
                             rotation=False,
                             examples_type=examples_type,
                             no_yellow=no_yellow,
                             model_dir=model_dir,
                             eval=eval)
    train_indices, val_indices = dataset.dataExtract.train_test_split(eval=True)
    if sampler_mode == 'all':
        indices = list(train_indices) + list(val_indices)
    else:
        indices = val_indices if sampler_mode == 'val' else train_indices
    if verbose > 1:
        print(f'Val indices length: {len(val_indices)}')
        print(f'Train indices length: {len(train_indices)}')
    if verbose > 2:
        print(np.array(dataset.group_names)[train_indices])
        print(np.array(dataset.group_names)[val_indices])
    return DataLoader(dataset, batch_size=len(val_indices), sampler=SubsetRandomSampler(indices), num_workers=8)


def load_model(model_path, pretrained_model_path, examples_type, fc_in_features, mode, explain):
    explain_component = get_explain_component(examples_type)
    if verbose > 2:
        print(explain_component)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if explain:
        model = ExpMVCNN(fc_in_features=fc_in_features,
                         pretrained_path=pretrained_model_path,
                         mode=mode,
                         explain_component=explain_component)
    else:
        model = MVCNN(fc_in_features=fc_in_features, mode=mode)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model, device