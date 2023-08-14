import os
import sys
import pandas
from os.path import dirname, abspath

project_path = dirname(dirname(abspath(__file__)))
model2_path = os.path.join(project_path, 'model2')
sys.path.append(project_path)
sys.path.append(model2_path)

import pdb
import pickle
import numpy as np
import pandas as pd

import shap
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from model2 import ObjectsDataset, MVCNN, ExpMVCNN, get_explain_component, verbose
from load import load_data, load_model, CPU_Unpickler, model_path_extractor, folder_dir_config
from config import preprocess_dir, verbose


examples_by_views = ['V1', 'V1', 'V2', 'V2', 'V3', 'V3', 'V4', 'V4', 'V5', 'V5']
# ---Model settings---
model_by_type = 'loss'   # loss / acc
fc_in_features = 128  # 64 / 128 / 256
num_workers = 8
# ---Model settings---

cur_date = '30_01_2023'
data_path = preprocess_dir  # directory of the data set after pre-process
explain = True
full_data_use = True
multiple_imgs = False

folder_dir = folder_dir_config(fc_in_features, cur_date, explain)


def batch_predict(exp_model, exp_images, pretrained_model):
    all_views_images = torch.stack([dataset_all.group_by_group(group_name)[0] for group_name in group_names])
    exp_outputs = exp_model(exp_images)
    all_outputs = pretrained_model(all_views_images)
    _, exp_preds = torch.max(exp_outputs, 1)
    _, all_preds = torch.max(all_outputs, 1)
    exp_preds, all_preds = exp_preds.cpu(), all_preds.cpu()
    # df = pd.DataFrame({'Model Predictions': all_preds, 'Labels': labels, 'Explainability Predictions': exp_preds})
    # df.to_csv(f'{save_dir}/SHAP_output_{n_test_images}.csv')
    return exp_preds, all_preds


def plot_graph(save_dir, n_test_images):
    save_dir = os.path.join(save_dir, plot_folder_name)
    with open(os.path.join(save_dir, f'shap_values_{n_test_images}.pkl'), 'rb') as f:
        shap_values = CPU_Unpickler(f).load()
    with open(os.path.join(save_dir, f'test_images_{n_test_images}.pkl'), 'rb') as f:
        test_images = CPU_Unpickler(f).load()
    test_numpy = reshape_visualization(test_images[:, 0, :, :, :, np.newaxis])
    shap_numpy = reshape_visualization(np.array(shap_values)[1, :, 0, :, :, :, np.newaxis])
    shap.image_plot(shap_numpy, -test_numpy)


def load_pretrained_model():
    dataset_all = ObjectsDataset(data_path=data_path,
                                 multiview=True,
                                 augmentation=False,
                                 rotation=False,
                                 examples_type='all',
                                 no_yellow=no_yellow,
                                 model_dir=model_dir,
                                 eval=True)
    pretrained_model = MVCNN(fc_in_features=fc_in_features, mode=3)
    pretrained_model.load_state_dict(torch.load(pretrained_model_path))
    pretrained_model.eval()
    return pretrained_model, dataset_all


def size_transf():
    return transforms.Compose([transforms.Resize((224, 224))])


def to_tensor():
    return transforms.Compose([transforms.ToTensor()])


def save_shap():
    with open(os.path.join(save_dir, f'shap_values_{n_test_images}.pkl'), 'wb') as f:
        pickle.dump(shap_values, f)
    with open(os.path.join(save_dir, f'test_images_{n_test_images}.pkl'), 'wb') as f:
        pickle.dump(images, f)


def reshape_visualization(array):
    array = np.swapaxes(array, 1, -1)
    return np.array(array[:, 0, :, :, :])


model_name = 'multi_all_mode3'
folder_model_dir = os.path.join(folder_dir, f'{model_name}')

examples_type = 'X20_OV'
plot = False
no_yellow = True
multiview = False
eval_all_data = False
pretrained_model_dir = os.path.join(model2_path, f'models/fc_in_features_128_17_01_2023/{model_name}')
pretrained_model_path = model_path_extractor(os.path.join(pretrained_model_dir, f'all_{int(no_yellow)}'), model_by_type)

save_dir = os.path.join(folder_model_dir, f'{examples_type}_{int(no_yellow)}')
cur_model_path = model_path_extractor(save_dir, model_by_type)
model_dir = dirname(dirname(save_dir))
n_test_images = 7
sampler_mode = 'val'
save_dir += '/SHAP'
plot_folder_name = '7_example_val'

if plot:
    plot_graph(save_dir, n_test_images)
else:
    try:
        idx = max([int(dir_name.split('_')[0]) for dir_name in os.listdir(save_dir) if 'example' in dir_name])
    except:
        idx = 0
    idx += 1
    print(f'{os.path.basename(folder_model_dir)} {examples_type}_{int(no_yellow)}: SHAP example {idx} folder has been created...\n')
    save_dir = os.path.join(save_dir, f'{idx}_example_{sampler_mode}')
    os.mkdir(save_dir)
    explain_component = get_explain_component(examples_type)

    model, device = load_model(pretrained_model_path=pretrained_model_path,
                               model_path=cur_model_path,
                               examples_type=examples_type,
                               fc_in_features=fc_in_features,
                               mode=int(model_name[-1]),
                               explain=explain)

    data_loader = load_data(data_path=data_path,
                            model_dir=model_dir,
                            examples_type=examples_type,
                            multiview=multiview,
                            no_yellow=no_yellow,
                            eval=True,
                            sampler_mode=sampler_mode)
    pretrained_model, dataset_all = load_pretrained_model()
    print('\nLoading images batch...')
    batch = next(iter(data_loader))
    images, labels, group_names = batch
    images = images.to(device)
    print(f'Number of examples: {len(images)}')
    e = shap.DeepExplainer(model, images)

    slice_range = slice(len(images)-n_test_images, len(images))
    images, labels, group_names = images[slice_range], labels[slice_range], group_names[slice_range]
    exp_preds, all_preds = batch_predict(exp_model=model, exp_images=images, pretrained_model=pretrained_model)
    df = pd.DataFrame({'Group Name': group_names, 'Explainability Predictions': exp_preds, 'Model Predictions': all_preds, 'Labels': labels})
    df = df.replace(0, 'red')
    df = df.replace(1, 'green')
    df.to_csv(os.path.join(save_dir, 'SHAP_output.csv'))
    print(f'Explainability Outputs: {exp_preds} \n'
          f'Model Outputs: {all_preds}\n'
          f'Labels: {labels}')
    shap_values = e.shap_values(images)
    save_shap()
