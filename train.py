import argparse
import math

import timm.scheduler
import torch.optim
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

import separable_laa
from dataset import *
import prompters
from utils import *


class PromptTrainer(object):

    def __init__(self, config: dict):
        self.scaler = GradScaler()
        print('Start training prepare...')
        self.config = config
        self.epoch = 0
        self.start_epoch = 0
        self.total_epoch = self.config['epoch']
        self.base_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.device = torch.device('cuda' if self.config['use_cuda'] and torch.cuda.is_available() else 'cpu')
        set_random_seed(self.config['seed'])
        print(f"Using specific random seed: {self.config['seed']}")
        self.num_classes = self.config['num_classes']
        self.mask_prob = self.config['mask_prob']
        self.mask_type = self.config['mask_type']
        self.transformers = self.get_transformers()
        self.datasets = self.get_dataset()
        self.dataloader = self.get_dataloader()
        print("Building Model...")
        self.model = self.get_model().to(self.device)
        print("Building Model OK!")
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

        # build meters
        self.performance_meters = self.get_performance_meters()
        self.average_meters = self.get_average_meters()

        # timer
        self.timer = Timer()
        print('Training Preparation Done!')

    def get_performance_meters(self):
        return {
            'train': {
                metric: PerformanceMeter(higher_is_better=False if 'loss' in metric else True)
                for metric in ['acc', 'loss']
            },
            'val': {
                metric: PerformanceMeter() for metric in ['acc']
            },
            'val_first': {
                metric: PerformanceMeter() for metric in ['acc']
            }
        }

    def get_average_meters(self):
        meters = ['acc', 'loss']  # Reset every epoch. 'acc' is reused in train/val/val_first stage.
        return {
            meter: AverageMeter() for meter in meters
        }

    def reset_average_meters(self):
        for meter in self.average_meters:
            self.average_meters[meter].reset()

    def get_model(self):
        model = separable_laa.lightweight_model(num_classes=self.num_classes, extra_method=self.config['method'],
                                                meta_len=self.config['meta_len'])
        return model

    def get_prompter(self):
        return prompters.__dict__[args.method](args).to(self.device)

    def get_transformers(self):
        resize_reso = 256
        crop_reso = 224

        return {
            'common_aug': transforms.Compose([
                transforms.Resize((resize_reso, resize_reso)),
                transforms.RandomHorizontalFlip(),
            ]),
            'train_totensor': transforms.Compose([
                transforms.RandomCrop((crop_reso, crop_reso)),
                transforms.RandomRotation(degrees=30),
                # transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.RandomErasing(0.2),
                # transforms.ColorJitter(0.1, 0.1),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val_totensor': transforms.Compose([
                transforms.CenterCrop((crop_reso, crop_reso)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'None': None,
        }

    def get_dataset(self):
        splits = ['train', 'val']
        return {
            split: CUBDataset(
                self.config['data_dir'], transforms=self.transformers, mode=split,
            ) for split in splits
        }

    def get_dataloader(self):
        splits = ['train', 'val']
        dataloaders = {
            split: DataLoader(
                self.datasets[split],
                self.config[split + '_batch_size'], num_workers=self.config['num_workers'], pin_memory=True,
                shuffle=split == 'train',
                drop_last=False,
            ) for split in splits
        }
        return dataloaders

    def get_criterion(self):
        return self.base_criterion

    def get_optimizer(self):
        params = [{'params': self.model.parameters()}, ]
        optimizer = torch.optim.AdamW(params, lr=self.config['lr'])
        return optimizer

    def get_scheduler(self):
        return timm.scheduler.CosineLRScheduler(self.optimizer, t_initial=self.config['epoch'], lr_min=1e-5,
                                                warmup_t=5, warmup_lr_init=1e-5)

    def train(self):
        self.model.train()
        # self.prompter.train()
        train_acc_l = []
        val_acc_l = []
        train_loss_l = []
        for epoch in range(self.start_epoch, self.total_epoch):
            self.epoch = epoch
            self.reset_average_meters()
            print(f"\nStarting epoch {epoch + 1}...")
            self.timer.tick()
            training_bar = tqdm(self.dataloader['train'], ncols=100)
            for data in training_bar:
                self.batch_training(data)
                training_bar.set_description(f'Train Epoch [{self.epoch + 1}/{self.total_epoch}]')
                training_bar.set_postfix(acc=self.average_meters['acc'].avg, loss=self.average_meters['loss'].avg)
            train_acc_l.append(self.average_meters['acc'].avg)
            train_loss_l.append(self.average_meters['loss'].avg)
            duration = self.timer.tick()
            print(f'Training duration {duration:.2f}s!')
            self.update_performance_meter('train')
            print(f'Starting validation stage in epoch {epoch + 1} ...')
            self.timer.tick()
            # validate
            self.validate()
            duration = self.timer.tick()
            print(f'Validation duration {duration:.2f}s!')

            # val stage metrics
            val_acc = self.average_meters['acc'].avg
            val_acc_l.append(val_acc)
            if self.performance_meters['val']['acc'].best_value is not None:
                is_best = epoch >= 0.5 * self.total_epoch and val_acc > self.performance_meters['val']['acc'].best_value
            else:
                is_best = epoch >= 0.5 * self.total_epoch
            self.update_performance_meter('val')
            self.do_scheduler_step()
            print(f'Epoch {epoch + 1} Done!')
            # save model
            if is_best:
                print('Saving best model ...')
                self.save_model()

        print(f'best acc:{self.performance_meters["val"]["acc"].best_value}')
        import pandas as pd
        data = pd.DataFrame({"train-acc": train_acc_l, "train-loss": train_loss_l, "val-acc": val_acc_l})
        data.to_csv("laa-meta-middis.csv")
        print("Training Done!")

    def mask_meta(self, meta):
        if self.mask_type == 'linear':
            cur_mask_prob = self.mask_prob * (1 - self.epoch / self.total_epoch)
        elif self.mask_type == "cosine":
            cur_mask_prob = self.mask_prob * (0.5 * (math.cos(self.epoch / self.total_epoch * math.pi) + 1))
        else:
            cur_mask_prob = self.mask_prob
        if cur_mask_prob != 0:
            mask = torch.ones_like(meta)
            mask_index = torch.randperm(meta.size(1))[:int(meta.size(1) * cur_mask_prob)]
            mask[:, mask_index] = 0
            meta = mask * meta
        return meta

    def batch_training(self, data):
        inputs, labels, metas = data

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        metas = metas.to(self.device)
        metas = self.mask_meta(metas)
        self.optimizer.zero_grad()
        # 半精计算
        with autocast():
            # forward
            # inputs = self.prompter(inputs, metas)
            outputs = self.model(inputs, metas)
            loss = self.criterion(outputs, labels)
        acc = accuracy(outputs, labels, 1)

        # backward

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # loss.backward()
        # self.optimizer.step()
        # record accuracy and loss
        self.average_meters['acc'].update(acc, labels.size(0))
        self.average_meters['loss'].update(loss.item(), labels.size(0))

    def validate(self):
        # self.prompter.train(False)
        self.model.train(False)
        self.reset_average_meters()

        with torch.no_grad():
            val_bar = tqdm(self.dataloader['val'], ncols=80)
            for data in val_bar:
                self.batch_validate(data)
                val_bar.set_description(f'Val Epoch [{self.epoch + 1}/{self.total_epoch}]')
                val_bar.set_postfix(acc=self.average_meters['acc'].avg)
        # self.prompter.train(True)
        self.model.train(True)

    def batch_validate(self, data):
        inputs = data[0].to(self.device)
        labels = data[1].long().to(self.device)
        metas = data[2].to(self.device)

        # forward
        with autocast():
            # forward
            # inputs = self.prompter(inputs, metas)
            outputs = self.model(inputs, metas)
        acc = accuracy(outputs, labels, 1)

        self.average_meters['acc'].update(acc, labels.size(0))

    def do_scheduler_step(self):
        self.scheduler.step(self.epoch + 1)

    def update_performance_meter(self, split):
        if split == 'train':
            self.performance_meters['train']['acc'].update(self.average_meters['acc'].avg)
            self.performance_meters['train']['loss'].update(self.average_meters['loss'].avg)
        elif split == 'val':
            self.performance_meters['val']['acc'].update(self.average_meters['acc'].avg)

    def save_model(self):
        output_model_path = './saved_models/'
        os.makedirs(output_model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_model_path, "saa-eip.pth"))
        print('model saved to: ', output_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "--pretrained", default=False, type=lambda x: x.lower() == 'true', help="use pretrained model (default:False)",
    )
    parser.add_argument(
        "--pretrained-model", type=str, default='', help="saved model path",
    )
    parser.add_argument(
        "--data-dir", type=str, default='./datasets', help="data path",
    )
    parser.add_argument(
        "--num-classes", type=int, default=200, help="num classes (default:200)",
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=64, help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=256, help="input batch size for testing (default: 256)",
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="num workers (default: 1)",
    )
    parser.add_argument(
        "--use-cuda", type=lambda x: x.lower() == 'true', default=True, help="enable CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    # meta config
    parser.add_argument(
        "--method", type=str, default="InsidePadMeta", help="prompt method"
    )
    parser.add_argument(
        "--meta-len", type=int, default=312,
    )
    parser.add_argument(
        "--mask-type", type=str, default="linear",
    )
    parser.add_argument(
        "--mask-prob", type=float, default=1.,
    )
    args = parser.parse_args()
    config = vars(args)
    trainer = PromptTrainer(config)
    trainer.train()
