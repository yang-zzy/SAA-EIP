# -*- coding:utf-8 -*-
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils import data


class ModelDataset(data.Dataset):
    def __init__(self, root, meta_path, transforms=None, mode='train'):
        super().__init__()
        self.root = root

        # load data
        data = pd.read_csv(meta_path, sep=' ', names=['label', 'path'], encoding='gbk')
        self.paths = data['path'].tolist()
        self.labels = data['label'].tolist()

        if mode == 'val':
            self.paths, self.labels = random_sample(self.paths, self.labels)
        self.common_aug = transforms['common_aug']
        self.totensor = transforms[mode + '_totensor']

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root, self.paths[item])
        img = self.pil_loader(img_path)
        img = self.common_aug(img) if not self.common_aug is None else img
        img = self.totensor(img)
        label = self.labels[item]
        return img, label, self.paths[item]

    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class CUBDataset(data.Dataset):
    def __init__(self, root, mode='train', transforms=None):

        self.root = root
        self.is_train = mode == 'train'
        self.common_aug = transforms['common_aug']
        self.totensor = transforms[mode + '_totensor']
        self.images_path = {}
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, root = line.split()
                self.images_path[image_id] = root

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id
        self.metas = []
        data = []
        with open(os.path.join(self.root, 'attributes/image_attribute_labels.txt'), 'r') as f:
            for line in f:
                line_data = line.strip().split()
                line_data = [int(x) if i < 2 else float(x) for i, x in enumerate(line_data)]
                data.append(line_data)
        for img in range(len(self.class_ids)):
            processed_line = []
            for attr in range(312):
                line = data[img * 312 + attr]
                assert line[0] == img + 1 and line[1] == attr + 1
                attribute_value = (line[2] - 0.5) * line[3] * line[4]
                processed_line.append(attribute_value)
            self.metas.append(processed_line)
        self.data_id = []
        if self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        self.data_id.append(image_id)
        if not self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        with open(os.path.join(self.root, 'images', path), 'rb') as f:
            with Image.open(f) as img:
                image = img.convert('RGB')

        if self.common_aug:
            image = self.common_aug(image)

        image = self.totensor(image)
        return image, class_id, torch.tensor(self.metas[int(image_id) - 1])

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]

    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]


def random_sample(img_names, labels):
    anno_dict = {}
    img_list = []
    anno_list = []
    for img, anno in zip(img_names, labels):
        if not anno in anno_dict:
            anno_dict[anno] = [img]
        else:
            anno_dict[anno].append(img)

    for anno in anno_dict.keys():
        anno_len = len(anno_dict[anno])
        fetch_keys = random.sample(list(range(anno_len)), anno_len // 10)
        img_list.extend([anno_dict[anno][x] for x in fetch_keys])
        anno_list.extend([anno for x in fetch_keys])
    return img_list, anno_list


class GrainSetDataset(data.Dataset):

    def __init__(self, root, mode='train', transforms=None):
        self.root = root
        self.common_aug = transforms['common_aug']
        self.totensor = transforms[mode + '_totensor']
        meta_path = os.path.join(root, mode + ".txt")
        # load data
        data = pd.read_csv(meta_path, sep=' ', names=['label', 'path', "meta"], encoding='gbk')
        self.paths = data['path'].tolist()
        self.labels = data['label'].tolist()
        self.metas = data["meta"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root, self.paths[item])
        img = self.pil_loader(img_path)
        img = self.common_aug(img) if not self.common_aug is None else img
        img = self.totensor(img)
        label = self.labels[item]
        meta = list(map(float, self.metas[item].split(",")))
        return img, label, torch.tensor(meta)

    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class DeepFashionDataset(data.Dataset):
    def __init__(self, root, mode='train', transforms=None):
        self.root = root
        self.common_aug = transforms['common_aug']
        self.totensor = transforms[mode + '_totensor']
        self.Anno = os.path.join(self.root, "Anno_fine")
        self.images_path = []
        with open(os.path.join(self.Anno, mode + '.txt')) as f:
            for line in f:
                image_path = line.split()[0]
                self.images_path.append(image_path)
        self.class_ids = []
        with open(os.path.join(self.Anno, mode + '_cate.txt')) as f:
            for line in f:
                class_id = line.split()[0]
                self.class_ids.append(class_id)
        self.metas = []
        with open(os.path.join(self.Anno, mode + '_attr.txt')) as f:
            for meta in f:
                metalist = list(map(float, meta[:-2].split(" ")))
                self.metas.append(metalist)
        assert len(self.images_path) == len(self.class_ids) == len(self.metas)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        path = self.images_path[item]
        with open(os.path.join(self.root, path), 'rb') as f:
            with Image.open(f) as img:
                image = img.convert('RGB')
        if self.common_aug:
            image = self.common_aug(image)
        image = self.totensor(image)
        class_id = self.class_ids[item]
        meta = self.metas[item]
        return image, class_id, torch.tensor(meta)
