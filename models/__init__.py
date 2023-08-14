import numpy as np
import io
import os
import random
import pdb
import torch
import pickle
import torch.nn as nn

from torch.utils.data import Dataset
from os.path import dirname, abspath
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from data_extract import DataExtract
from config import models2_dir, verbose


project_path = dirname(dirname(abspath(__file__)))
multiview_arr = ['all', 'X10', 'X20']
resnet_cut_idx = {64: -5, 128: -4, 256: -3}
# ---Model settings---
model_by_type = 'loss'  # loss / acc
fc_in_features = 128  # 64 / 128 / 256
num_workers = 8
# ---Model settings---


def get_explain_component(examples_type):
    if examples_type in ['V1', 'V2', 'V3', 'V4', 'V5']:
        explain_component = examples_type
    else:
        explain_component = 'profiles' if 'X20' in examples_type else 'topbottom'
    return explain_component


def normalize(img):
    return np.float32(np.array(img) / 255)


def noise(image, grayscale):
    image = np.array(image)
    if grayscale:
        ch = 1
        row, col = image.shape
    else:
        row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.round(np.random.normal(mean, sigma, (row, col, ch)) * 10)
    gauss = gauss.reshape(row, col, ch)
    if grayscale:
        return np.uint8(abs(image + gauss[:, :, 0]))
    return np.uint8(abs(image + gauss))


def convert_to_bins(img, bins):
    img = np.array(img)
    jump = int(255/bins)
    for min_val in range(0, 255-2*jump, jump):
        max_val = min_val + jump
        img[(img >= min_val) & (img <= max_val)] = min_val
    min_val += jump
    img[(img >= min_val)] = min_val
    return img


def rotation_angles(filename):
    if int(filename.split('-')[-1][0]) <= 1:
        return list(np.arange(0, 360, 30))
    return [-10, -5, 0, 5, 10]


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = int(random.choice(self.angles))
        return TF.rotate(x, angle)


class ObjectsDataset(Dataset):
    def __init__(self, data_path,
                 multiview=True,
                 augmentation=False,
                 rotation=False,
                 examples_type='all',
                 no_yellow=False,
                 model_dir=None,
                 eval=False,
                 base_train_test=False):
        self.data_path = data_path
        self.examples_type = examples_type
        self.augmentation = augmentation
        self.rotation = rotation
        self.dataExtract = DataExtract(data_path=data_path,
                                       multiview=multiview,
                                       examples_type=examples_type,
                                       no_yellow=no_yellow,
                                       model_dir=model_dir,
                                       eval=eval,
                                       base_train_test=base_train_test)
        self.group_names = self.dataExtract.group_names
        self.group_labels = self.dataExtract.group_labels
        self.group_labels_idx = self.dataExtract.group_labels_idx
        self.outer_group_names = self.dataExtract.outer_group_names
        if verbose > 1:
            print(f'number of labels: {len(self.group_labels)} \nnumber of outer groups: {len(self.outer_group_names)}')
        if verbose > 2:
            print(f'{self.group_names[0]}\n {self.group_names[1]} \n {self.group_names[2]} \n {self.group_names[3]}')

    def __len__(self):
        return len(self.group_names)

    def _transform(self, image, rotation_angles):
        resize = transforms.Compose([transforms.Resize((224, 224))])
        to_tensor = transforms.Compose([transforms.ToTensor()])
        image = resize(image)
        if not self.augmentation:
            image = Image.fromarray(np.array(image)/255.0)
            return to_tensor(image)

        noisy_image = Image.fromarray(noise(image, grayscale=True))
        color_transform = transforms.Compose([transforms.ColorJitter([0.8, 1.5], [0.8, 1.2], [0.8, 1.2], [-0.1, 0.1])])
        if self.rotation:
            image_rotation = MyRotationTransform(angles=rotation_angles)
            return to_tensor(normalize(image_rotation(color_transform(noisy_image))))
        else:
            return to_tensor(normalize(color_transform(noisy_image)))

    def stack_group(self, group_fnames, item_path):
        if self.examples_type == 'X10':
            return torch.stack([self._transform(Image.open(os.path.join(item_path, fname)).convert('L'), rotation_angles(fname))
                                for fname in group_fnames if int(fname.split('-')[3][0]) < 2])
        if self.examples_type == 'X20':
            return torch.stack([self._transform(Image.open(os.path.join(item_path, fname)).convert('L'), rotation_angles(fname))
                                for fname in group_fnames if int(fname.split('-')[3][0]) > 1])
        return torch.stack([self._transform(Image.open(os.path.join(item_path, fname)).convert('L'), rotation_angles(fname))
                            for fname in group_fnames])

    def group_by_group(self, group_name):
        label = self.group_labels[np.where(self.group_names == group_name)[0]]
        group_fnames = [fname for fname in os.listdir(self.data_path) if fname.startswith(group_name)][:5]
        group = self.stack_group(group_fnames, self.data_path)
        return group, label

    def __getitem__(self, index):
        group_name = self.group_names[index]
        label = self.group_labels[index]
        # Get Images of the group
        group_fnames = [fname for fname in os.listdir(self.data_path) if fname.startswith(group_name)][:5]
        group = self.stack_group(group_fnames, self.data_path)
        global_group_name = group_name if not '.png' in group_name else '-'.join(group_name.split('-')[:3])
        return group, label, global_group_name


# Return the number of learnable parameters for a given model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MVCNN(nn.Module):
    def __init__(self, fc_in_features=128, mode=1):
        super(MVCNN, self).__init__()
        self.fc_in_features = fc_in_features
        self.mode = mode
        if mode == 1:
            self.fe = self.mode1_init()
        if mode == 2:
            self.fe1, self.fe2, self.fe3, self.fe4, self.fe5 = self.mode2_init()
        if mode == 3:
            self.fe_profiles, self.fe_topbottom = self.mode3_init()
        if mode == 4:
            self.fe1, self.fe2, self.fe3, self.fe4, self.fe5 = self.mode2_init()
            self.cl1 = self.classifier_init(fc_in_features=fc_in_features)
            self.cl2 = self.classifier_init(fc_in_features=fc_in_features)
            self.classifier = self.classifier_init(fc_in_features=4)

        if mode != 4:
            self.classifier = self.classifier_init(fc_in_features=fc_in_features)

    def resnet_init(self, fc_in_features):
        resnet = models.resnet34(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        return list(resnet.children())[:resnet_cut_idx[fc_in_features]]

    def classifier_init(self, fc_in_features):
        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_in_features, 2)
        )
        return classifier

    def mode1_init(self):
        self.params = self.resnet_init(self.fc_in_features)
        return nn.Sequential(*self.params, nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def mode2_init(self):
        self.params = [self.resnet_init(self.fc_in_features) for i in range(5)]
        self.features = nn.Sequential(*self.params[0], nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                      *self.params[1], nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                      *self.params[2], nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                      *self.params[3], nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                      *self.params[4], nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        return [self.features[i:i+7] for i in range(0, 35, 7)]

    def mode3_init(self):
        self.params = [self.resnet_init(self.fc_in_features) for i in range(2)]
        self.features = nn.Sequential(*self.params[0], nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                      *self.params[1], nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        return self.features[:7], self.features[7:]

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # -> inputs.shape = views x samples x channels x height x width
        view_features = []

        if self.mode == 1:
            for view_batch in inputs:
                view_batch = self.fe(view_batch)
                view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
                view_features.append(view_batch)

        if self.mode == 2:
            for view_batch, fe in zip(inputs, [self.fe1, self.fe2, self.fe3, self.fe4, self.fe5]):
                view_batch = fe(view_batch)
                view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
                view_features.append(view_batch)

        if self.mode == 3:
            for i, view_batch in enumerate(inputs):
                if i < 2:
                    view_batch = self.fe_topbottom(view_batch)
                else:
                    view_batch = self.fe_profiles(view_batch)
                view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
                view_features.append(view_batch)

        if self.mode == 4:
            for view_batch, fe in zip(inputs, [self.fe1, self.fe2, self.fe3, self.fe4, self.fe5]):
                view_batch = fe(view_batch)
                view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
                view_features.append(view_batch)
        else:
            pooled_views, _ = torch.max(torch.stack(view_features), 0)
            outputs = self.classifier(pooled_views)
            return outputs

        pooled_views_topbottom, _ = torch.max(torch.stack(view_features[:2]), 0)
        pooled_views_profiles, _ = torch.max(torch.stack(view_features[2:]), 0)
        output1 = self.cl1(pooled_views_topbottom)
        output2 = self.cl2(pooled_views_profiles)
        return self.classifier(torch.concat((output1, output2), dim=1))


class ExpMVCNN(nn.Module):
    def __init__(self, fc_in_features=128, pretrained_path=None, mode=3, explain_component='profiles'):
        super(ExpMVCNN, self).__init__()
        self.mode = mode
        self.explain_component = explain_component
        pretrained_model = MVCNN(fc_in_features=fc_in_features, mode=mode)
        pretrained_model.load_state_dict(torch.load(pretrained_path))
        if mode == 2:
            self.fe = self.mode2_init(pretrained_model)
        if mode == 3:
            self.fe = self.mode3_init(pretrained_model)
        if mode == 4:
            self.fe = self.mode2_init(pretrained_model)
        self.classifier = self.classifier_init(fc_in_features=fc_in_features)

    def classifier_init(self, fc_in_features):
        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_in_features, 2)
        )
        return classifier

    def mode2_init(self, pretrained_model):
        if self.explain_component == 'V1':
            return pretrained_model.fe1
        if self.explain_component == 'V2':
            return pretrained_model.fe2
        if self.explain_component == 'V3':
            return pretrained_model.fe3
        if self.explain_component == 'V4':
            return pretrained_model.fe4
        if self.explain_component == 'V5':
            return pretrained_model.fe5

    def mode3_init(self, pretrained_model):
        return pretrained_model.fe_profiles if self.explain_component == 'profiles' else pretrained_model.fe_topbottom

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # -> inputs.shape = views x samples x channels x height x width
        view_batch = self.fe(inputs[0])
        view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
        outputs = self.classifier(view_batch)
        return outputs