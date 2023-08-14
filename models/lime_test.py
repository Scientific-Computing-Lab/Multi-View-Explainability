import os
import sys
from os.path import dirname, abspath

project_path = dirname(dirname(abspath(__file__)))
model2_path = os.path.join(project_path, 'model2')
sys.path.append(project_path)
sys.path.append(model2_path)

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import pdb

from model2 import MVCNN, ExpMVCNN, get_explain_component, model_dir_config, verbose
from config import preprocess_dir, verbose


models_names = ['bottom', 'top', 'top_bottom']
models_names = ['multi_all_mode2', 'multi_all_mode3', 'multi_all_mode4']

# Examples type:
# Multiview: all, X10, X20
# Seperated:  X10_0 (bottom), X10_1 (top), X10_both
examples_by_views = ['V1', 'V1', 'V2', 'V2', 'V3', 'V3', 'V4', 'V4', 'V5', 'V5']
examples_types = [['X10_0', 'X10_0'], ['X10_1', 'X10_1'], ['X10_both', 'X10_both']]
examples_types = [examples_by_views, ['X10_OV', 'X10_OV', 'X20_OV', 'X20_OV'], examples_by_views]

# ---Model settings---
model_by_type = 'loss'   # loss / acc
fc_in_features = 128  # 64 / 128 / 256
num_workers = 8
# ---Model settings---

cur_date = '30_01_2023'
images_dir = preprocess_dir
explain = True
full_data_use = True
multiple_imgs = False

model_dir = model_dir_config(fc_in_features, cur_date, full_data_use)
if explain:
    model_dir = f'{model_dir}_ex'
if verbose > 0:
    print(model_dir)
if not multiple_imgs:
    img_name = 'T479-2-9-2'  # Profile:T479-2-9-2  Profile:T489-2-8-4  Profile:T479-2-10-2


def model_path_extractor(path):
    return [os.path.join(path, fname) for fname in os.listdir(path) if fname.startswith(f'model_by_{model_by_type}')][0]


def plot_graphs(img_name, params, save_dir):
    img_name = img_name.split('.')[0]
    for key, value in params.items():
        plt.imshow(value)
        plt.axis('off')
        if key=='temp':
            plt.savefig(os.path.join(save_dir, f'{img_name}.png'))
        plt.close()


def load_model(cur_model_dir, explain_component, pretrained_model_path, mode):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if explain:
        model = ExpMVCNN(fc_in_features=fc_in_features,
                         pretrained_path=pretrained_model_path,
                         mode=mode,
                         explain_component=explain_component)
    else:
        model = MVCNN(fc_in_features=fc_in_features, mode=mode)
    model.load_state_dict(torch.load(cur_model_dir))
    model.eval()
    model.to(device)
    return model, device


def color_replace(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_threshold = {'low': np.array([0, 0, 0]), 'high': np.array([255, 255, 255])}
    mask = cv2.inRange(hsv, color_threshold['low'], color_threshold['high'])
    hsv[mask > 0, 0] = 110
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_image(path, color=False):
    if color:
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    return Image.open(path).convert('L')


def size_transf():
    return transforms.Compose([transforms.Resize((224, 224))])


def to_tensor():
    return transforms.Compose([transforms.ToTensor()])


def batch_predict(images, verbose=0, lime=True):
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0).unsqueeze(0)
    if lime:
        batch = batch[0, :, 0, np.newaxis, :, :]
        batch = batch[:, np.newaxis, :, :, :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device).to(torch.float32)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    if verbose > 1:
        print(f'batch shape: {batch.shape}')
    if verbose > 2:
        print(probs)
    return probs.detach().cpu().numpy()


def prediction_explain(img):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)).astype('double'),
                                             batch_predict,  # classification function
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)  # number of images that will be sent to classification function
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
    return temp, mask


def single_prediction():
    test_pred = batch_predict([np.array(pill_transf(img))], lime=False)
    print(test_pred)
    print(test_pred.squeeze().argmax())


def run(img, img_name):
    global model, device
    for i, model_name in enumerate(models_names):
        pretrained_model_dir = os.path.join(model2_path, f'models/fc_in_features_128_17_01_2023/{model_name}')
        folder_dir = os.path.join(model_dir, f'{model_name}')
        for j, examples_type in enumerate(examples_types[i]):
            no_yellow = False
            if (j + 1) % 2 == 0:
                no_yellow = True
            save_dir = os.path.join(folder_dir, f'{examples_type}_{int(no_yellow)}')
            # for fname in os.listdir(save_dir):
            #     img = cv2.imread(os.path.join(save_dir, fname))
            #     img = color_replace(img)
            #     cv2.imwrite(os.path.join(f'{save_dir}_attention', fname), img)
            if explain:
                pretrained_model_path = model_path_extractor(os.path.join(pretrained_model_dir, f'all_{int(no_yellow)}'))
                cur_model_dir = model_path_extractor(save_dir)
            explain_component = get_explain_component(examples_type, examples_by_views)
            model, device = load_model(pretrained_model_path=pretrained_model_path,
                                       cur_model_dir=cur_model_dir,
                                       explain_component=explain_component,
                                       mode=int(model_name[-1]))
            torch.manual_seed(1)
            for img_name in os.listdir(images_dir):
                img_view = int(img_name.split('.')[0][-1])
                fit = False
                if explain_component == 'profiles':
                    if img_view >= 2:
                        fit = True
                elif explain_component == 'topbottom':
                    if img_view < 2:
                        fit = True
                elif img_view == int(explain_component[-1])-1:
                    fit = True

                if fit:
                    img = get_image(path=os.path.join(images_dir, img_name))
                    img = Image.fromarray(np.array(img) / 255.0)
                    temp, mask = prediction_explain(img.copy())
                    img_boundary = mark_boundaries(temp / 255.0, mask)
                    params = {'mask': mask, 'temp': temp, 'marked_image': img_boundary}
                    plot_graphs(img_name, params, os.path.join(save_dir, 'LIME'))


def lime_folders_open():
    for i, folder in enumerate(models_names):
        folder_dir = os.path.join(model_dir, f'{folder}')
        for j, examples_type in enumerate(examples_types[i]):
            no_yellow = False
            if (j + 1) % 2 == 0:
                no_yellow = True
            try:
                os.mkdir(os.path.join(os.path.join(folder_dir, f'{examples_type}_{int(no_yellow)}'), 'LIME_attention'))
            except:
                if verbose > 0:
                    print(f'{folder} {examples_type}_{j}: LIME folder is already exists')


pill_transf = size_transf()
preprocess_transform = to_tensor()
if multiple_imgs:
    lime_folders_open()
    for img_name in os.listdir(images_dir):
        if int(img_name.split('.')[0][-1]) < 2:
            img = get_image(path=os.path.join(images_dir, img_name))
            img = Image.fromarray(np.array(img) / 255.0)
            run(img, img_name)
else:
    lime_folders_open()
    img = get_image(path=os.path.join(images_dir, f'{img_name}.png'))
    img = Image.fromarray(np.array(img) / 255.0)
    run(img, img_name)

