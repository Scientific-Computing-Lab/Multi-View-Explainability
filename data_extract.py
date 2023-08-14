import numpy as np
import pandas as pd
import pickle
import os
import pdb

from os.path import dirname

from config import data_dir, verbose


class DataExtract:
    def __init__(self,
                 data_path,
                 multiview=True,
                 examples_type='all',
                 no_yellow=False,
                 test_size=0.2,
                 model_dir=None,
                 eval=eval,
                 base_train_test=False):
        self.data_path = data_path
        self.multiview = multiview
        self.examples_type = examples_type
        self.no_yellow = no_yellow
        self.model_dir = model_dir
        print(self.model_dir)
        self.test_size = test_size
        self.eval = eval
        self.base_train_test = base_train_test
        self.classes_id = {'red': 0,
                           'yellow': 0,
                           'green': 1}
        self.inner_group_size = 1
        self.examples_by_view = ['V1', 'V2', 'V3', 'V4', 'V5']
        if self.examples_type == 'X10_OV':
            self.inner_group_size = 2
        if self.examples_type == 'X20_OV':
            self.inner_group_size = 3
        self.outer_group_names = np.unique(['-'.join(fname.split('-')[:3]) for fname in os.listdir(self.data_path)])
        self.labels = self.get_labels()
        self.group_names, self.group_labels, self.group_labels_idx = self.group_details(self.labels)
        if verbose > 1:
            print(f'number of groups: {len(self.group_names)}')

    def idxs_load(self):
        with open(os.path.join(data_dir, f'idxs_split.pkl'), 'rb') as f:
            return pickle.load(f)

    def inner_group_names(self, group_fnames):
        if self.examples_type in self.examples_by_view:
            view_idx = int(self.examples_type[-1]) - 1
            return [fname for fname in group_fnames if int(fname.split('-')[-1][0]) == view_idx]
        if self.examples_type == 'X10_OV':
            return [fname for fname in group_fnames if int(fname.split('-')[-1][0]) < 2]
        if self.examples_type == 'X20_OV':
            return [fname for fname in group_fnames if int(fname.split('-')[-1][0]) >= 2]
        return [fname for fname in group_fnames if fname.split('-')[3][0] == self.examples_type[-1]]

    def group_details(self, labels):
        if self.multiview:
            if verbose > 2:
                print(f'outer group names: {self.outer_group_names}')
            return self.outer_group_names, np.array([self.classes_id[label] for label in labels['label']]), np.array(labels.index)
        inner_group_names = []
        inner_group_labels = []
        inner_group_labels_idx = []
        group_names = np.array(labels['model_name'])
        for idx, group_name in enumerate(group_names):
            group_fnames = [fname for fname in os.listdir(self.data_path) if fname.startswith(group_name)]
            cur_inner_group_names = self.inner_group_names(group_fnames)
            inner_group_names += cur_inner_group_names
            inner_group_labels += [self.classes_id[labels[labels['model_name'] == group_name]['label'].values[0]]] * len(cur_inner_group_names)
            label_idx = labels.loc[labels['model_name'] == group_name].index[0]
            inner_group_labels_idx += list(np.arange(label_idx * self.inner_group_size, label_idx * self.inner_group_size + self.inner_group_size))
        return inner_group_names, np.array(inner_group_labels), np.array(inner_group_labels_idx)

    def get_labels(self):
        images_labels = pd.read_excel(f'{data_dir}/image_labels.xlsx').iloc[:, 1:3]
        labels = pd.merge(pd.DataFrame({'model_name': self.outer_group_names}), images_labels, on=['model_name'], how='left')
        labels.dropna(inplace=True)
        if self.no_yellow:
            labels = labels.loc[labels['label'] != 'yellow']
        labels.reset_index(inplace=True)
        self.outer_group_names = self.outer_group_names[labels['index']]
        return labels

    def convert_to_real_idxs(self, group_train_idx, group_test_idx):
        group_train_idx = [self.labels[self.labels['index'] == idx].index[0] for idx in group_train_idx
                           if len(self.labels[self.labels['index'] == idx].index) > 0]
        group_test_idx = [self.labels[self.labels['index'] == idx].index[0] for idx in group_test_idx
                          if len(self.labels[self.labels['index'] == idx].index) > 0]
        return np.array(group_train_idx), np.array(group_test_idx)

    def inner_group_idxs(self, group):
        group_idx = []
        for idx in group:
            group_idx += list(np.arange(idx * self.inner_group_size, idx * self.inner_group_size + self.inner_group_size))
        return group_idx

    def save_train_test(self, train, test, path):
        with open(f'{path}/train.pickle', 'wb') as f:
            pickle.dump(train, f)
        with open(f'{path}/test.pickle', 'wb') as f:
            pickle.dump(test, f)

    def write_info(self, train, test):
        info = ''
        n_per_label = {'train': {}, 'test': {}}
        for label in ('green', 'yellow', 'red'):
            n_per_label['train'][label] = len(train['label'].loc[train['label'] == label])
            n_per_label['test'][label] = len(test['label'].loc[test['label'] == label])
        for mode in ('train', 'test'):
            info += f"{mode} - green:{n_per_label[mode]['green']}, " \
                    f"yellow:{n_per_label[mode]['yellow']}, " \
                    f"red:{n_per_label[mode]['red']}\n"
        with open(os.path.join(self.model_dir, 'info.txt'), 'w') as f:
            f.write(info)

    def load_train_test(self, path):
        postfix = ''
        if path == data_dir:
            postfix = '_base'
        with open(f'{path}/train{postfix}.pickle', 'rb') as f:
            train = pickle.load(f)
        with open(f'{path}/test{postfix}.pickle', 'rb') as f:
            test = pickle.load(f)
        self.write_info(train, test)
        return train, test

    def update_idxs(self):
        train, test = self.load_train_test(data_dir)
        train = self.labels[self.labels['model_name'].isin(list(train['model_name']))]
        test = self.labels[self.labels['model_name'].isin(list(test['model_name']))]
        new_examples = self.labels[~self.labels['model_name'].isin(list(train['model_name'])+list(test['model_name']))]
        for label in ['green', 'yellow', 'red']:
            df = new_examples[new_examples['label']==label]
            train = pd.concat([train, df[0:round(len(df)/2)]])
            test = pd.concat([test, df[round(len(df)/2):]])
        self.save_train_test(train, test, data_dir)

    def ratio_handler(self, random=False):
        train, test = self.load_train_test(path=data_dir)
        if random:
            test = test.sample(frac=1)
        add_train = int((len(train) + len(test))*(1-self.test_size)) - len(train)
        train = pd.concat([train, test.iloc[:add_train]])
        test = test.iloc[add_train:]

        train.to_csv(os.path.join(self.model_dir, 'train.csv'))
        test.to_csv(os.path.join(self.model_dir, 'test.csv'))
        self.write_info(train, test)
        return train, test

    def load_idxs(self, eval=False):
        if self.base_train_test:
            train, test = self.load_train_test(data_dir)
            return np.array(train['index']), np.array(test['index'])

        if eval or os.path.isfile(f'{self.model_dir}/train.pickle'):
            train, test = self.load_train_test(self.model_dir)
            return np.array(train['index']), np.array(test['index'])

        train, test = self.ratio_handler(random=True)
        self.save_train_test(train, test, self.model_dir)
        return np.array(train['index']), np.array(test['index'])

    def train_test_split(self, eval=False):
        group_train_idx, group_test_idx = self.convert_to_real_idxs(*self.load_idxs(eval))
        if verbose > 2:
            print(f'group_train_idx: {group_train_idx} \n group_test_idx: {group_test_idx}')
        if self.examples_type == 'X10_OV' or self.examples_type == 'X20_OV':
            if verbose > 2:
                print(f'inner_group_size: {self.inner_group_size}')
            return self.inner_group_idxs(group_train_idx), self.inner_group_idxs(group_test_idx)
        return group_train_idx, group_test_idx