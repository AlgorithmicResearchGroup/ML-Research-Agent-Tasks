from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.loss import BinaryCrossEntropy
from timm.data.transforms_factory import create_transform
from timm.data import Mixup
from timm.data.config import resolve_data_config
import timm
import argparse
import random
from glob import glob
import yaml
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from random import seed, shuffle
from torchvision.datasets.folder import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import warnings
from sklearn import metrics
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import cv2
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import pathlib
import os
import time
from PIL import Image
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from datetime import datetime
import transformers
warnings.filterwarnings('ignore')


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IMAGENET100(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_id = self.df.iloc[index]['image_id']
        label = int(self.df.iloc[index]['class'])
        path = os.path.join(self.img_dir, f'{image_id}.jpeg')
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)


class IMAGENET100_test(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_id = self.img_paths[index].split('/')[-1].split('.')[0]
        path = self.img_paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_id


def get_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])


def read_yaml_string(yaml_string):
    try:
        data = yaml.safe_load(yaml_string)
        return data
    except yaml.YAMLError as e:
        print(f"Error while parsing YAML string: {e}")


def convert_dict_to_argparse(dictionary):
    parser = argparse.Namespace()
    for key, value in dictionary.items():
        setattr(parser, key, value)
    return parser


def get_train_val_test_dataset(root_dir, image_size, mean, std):
    train_df = pd.read_csv(os.path.join(root_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(root_dir, 'val.csv'))
    test_files = glob(f'{root_dir}/test/*')
    transform = get_transform(image_size, mean, std)
    train_dataset = IMAGENET100(
        train_df, os.path.join(root_dir, "train"), transform)
    val_dataset = IMAGENET100(val_df, os.path.join(root_dir, "val"), transform)
    test_dataset = IMAGENET100_test(test_files, transform)
    return (train_dataset, val_dataset, test_dataset)


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        crop_pct=None,
        tf_preprocessing=False,
        **kwargs
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )
    print(dataset.transform)
    loader = None
    if kwargs['data_type'] == 'train':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_training, num_workers=num_workers,drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_training, num_workers=num_workers)

    return loader


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        self.encoder = timm.create_model(
            args.model,
            pretrained=args.pretrained,
            in_chans=3,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint,
        )

    def forward(self, x):
        return self.encoder(x)


class Trainer():
    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 device,
                 model,
                 optimizer,
                 scheduler,
                 epochs,
                 model_save_dir,
                 train_criterion,
                 val_criterion,
                 mixup_fn,
                 test_output_dir,
                 train_time):

        self.epochs = epochs
        self.epoch = 0
        self.accelarator = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.test_output_dir = test_output_dir
        self.model_save_dir = model_save_dir

        self.train_loader = train_loader
        self.validation_loader = val_loader
        self.test_loader = test_loader

        self.mixup_fn = mixup_fn
        self.train_criterion = train_criterion
        self.val_criterion = val_criterion
        self.train_time = train_time
        self.train_start = -1
        self.train_end = -1

    def get_metrics(self, predictions, actual, isTensor=False):
        if isTensor:
            p = predictions.detach().cpu().numpy()
            a = actual.detach().cpu().numpy()
        else:
            p = predictions
            a = actual
        accuracy = metrics.accuracy_score(y_pred=p, y_true=a)
        return {
            "accuracy": accuracy
        }

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train_step(self):
        self.model.train()
        print("Train Loop!")
        running_loss_train = 0.0
        num_train = 0
        train_predictions = np.array([])
        train_labels = np.array([])

        for images, labels in tqdm(self.train_loader):
            images = images.to(self.accelarator)
            labels = labels.to(self.accelarator)
            images, labels = self.mixup_fn(images, labels)

            num_train += labels.shape[0]
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.train_criterion(outputs, labels)
            running_loss_train += loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.lr = self.get_lr(self.optimizer)

            if time.time() >= self.train_end:
                return TRAIN_END_FLAG

        print(f"Train Loss: {running_loss_train/num_train}")

        return {
            'loss': running_loss_train/num_train,
        }

    def val_step(self):
        val_predictions = np.array([])
        val_labels = np.array([])
        running_loss_val = 0.0
        num_val = 0
        self.model.eval()
        with torch.no_grad():
            print("Validation Loop!")
            for images, labels in tqdm(self.validation_loader):
                images = images.to(self.accelarator)
                labels = labels.to(self.accelarator)
                outputs = self.model(images)
                num_val += labels.shape[0]
                _, preds = torch.max(outputs, 1)
                val_predictions = np.concatenate(
                    (val_predictions, preds.detach().cpu().numpy()))
                val_labels = np.concatenate(
                    (val_labels, labels.detach().cpu().numpy()))

                loss = self.val_criterion(outputs, labels)
                running_loss_val += loss.item()
                if time.time() >= self.train_end:
                    return TRAIN_END_FLAG
            val_metrics = self.get_metrics(val_predictions, val_labels)
            print(f"Validation Loss: {running_loss_val/num_val}")
            print(f"Val Accuracy Metric: {val_metrics['accuracy']} ")
            return {
                'loss': running_loss_val/num_val,
                'accuracy': val_metrics['accuracy'],
            }

    def test_step(self):
        test_image_ids = np.array([])
        test_predictions = np.array([])
        best_model_path = os.path.join(self.model_save_dir,'best_val_accuracy.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(
                f"{self.model_save_dir}/best_val_accuracy.pt")
            start_epoch = checkpoint['epoch']
            print(
                f"Model already trained for {start_epoch} epochs.")
            print(self.model.load_state_dict(checkpoint['model1_weights']))
        self.model.eval()
        with torch.no_grad():
            print("Test Loop!")
            for images, image_ids in tqdm(self.test_loader):
                images = images.to(self.accelarator)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                test_image_ids = np.concatenate((test_image_ids, image_ids))
                test_predictions = np.concatenate(
                    (test_predictions, preds.detach().cpu().numpy()))

            pd.DataFrame({
                'image_id': test_image_ids,
                'label': test_predictions.astype(int)
            }).to_csv(os.path.join(self.test_output_dir, 'submission.csv'), index=False)

    def run(self, run_test=True):
        best_validation_loss = float('inf')
        best_validation_accuracy = 0
        self.train_start = time.time()
        self.train_end = self.train_start + self.train_time
        for epoch in range(self.epochs):
            print("="*31)
            print(f"{'-'*10} Epoch {epoch+1}/{self.epochs} {'-'*10}")
            train_logs = self.train_step()
            if train_logs == TRAIN_END_FLAG:
                torch.save({
                    'model1_weights': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'epoch': epoch+1,
                }, f"{self.model_save_dir}/last.pt")
                break
            val_logs = self.val_step()
            if val_logs == TRAIN_END_FLAG:
                torch.save({
                    'model1_weights': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'epoch': epoch+1,
                }, f"{self.model_save_dir}/last.pt")
                break
            self.epoch = epoch
            if val_logs["loss"] < best_validation_loss:
                best_validation_loss = val_logs["loss"]
                torch.save({
                    'model1_weights': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'epoch': epoch+1,
                }, f"{self.model_save_dir}/best_val_loss.pt")
            if val_logs['accuracy'] > best_validation_accuracy:
                best_validation_accuracy = val_logs['accuracy']
                torch.save({
                    'model1_weights': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'epoch': epoch+1,
                }, f"{self.model_save_dir}/best_val_accuracy.pt")
        if run_test:
            self.test_step()

        return {
            'best_accuracy': best_validation_accuracy,
            'best_loss': best_validation_loss,
        }


if __name__ == "__main__":
    yaml_string = '''
        aa: rand-m6-mstd0.5-inc1
        distributed: false
        device: 0
        rank: 0
        amp: true
        apex_amp: false
        aug_repeats: 0
        aug_splits: 0
        batch_size: 1024
        bce_loss: true
        bn_eps: null
        bn_momentum: null
        bn_tf: false
        channels_last: true
        checkpoint_hist: 10
        clip_grad: null
        clip_mode: norm
        color_jitter: 0.4
        cooldown_epochs: 10
        crop_pct: 0.95
        cutmix: 1.0
        cutmix_minmax: null
        data_dir: /imagenet
        decay_epochs: 100
        decay_rate: 0.1
        dist_bn: reduce
        drop: 0.0
        drop_block: null
        drop_connect: null
        drop_path: null
        epoch_repeats: 0.0
        epochs: 100
        eval_metric: top1
        experiment: ""
        gp: null
        hflip: 0.5
        img_size: 160
        initial_checkpoint: ""
        input_size: null
        interpolation: ""
        jsd_loss: false
        local_rank: 0
        log_interval: 50
        log_wandb: false
        lr: 0.008
        lr_cycle_decay: 0.5
        lr_cycle_limit: 1
        lr_cycle_mul: 1.0
        lr_k_decay: 1.0
        lr_noise: null
        lr_noise_pct: 0.67
        lr_noise_std: 1.0
        mean: null
        min_lr: 1.0e-06
        mixup: 0.1
        mixup_mode: batch
        mixup_off_epoch: 0
        mixup_prob: 1.0
        mixup_switch_prob: 0.5
        model: resnet50
        model_ema: false
        model_ema_decay: 0.9998
        model_ema_force_cpu: false
        momentum: 0.9
        native_amp: false
        no_aug: false
        no_prefetcher: true
        no_resume_opt: false
        num_classes: null
        opt: lamb
        opt_betas: null
        opt_eps: null
        output: ""
        patience_epochs: 10
        pin_mem: false
        pretrained: false
        ratio:
        - 0.75
        - 1.3333333333333333
        recount: 1
        recovery_interval: 0
        remode: pixel
        reprob: 0.0
        resplit: false
        resume: ""
        save_images: false
        scale:
        - 0.08
        - 1.0
        sched: cosine
        seed: 0
        smoothing: 0.0
        split_bn: false
        start_epoch: null
        std: null
        sync_bn: false
        torchscript: false
        train_interpolation: random
        train_split: train
        tta: 0
        use_multi_epochs_loader: false
        val_split: validation
        validation_batch_size: null
        vflip: 0.0
        warmup_epochs: 5
        warmup_lr: 0.0001
        weight_decay: 0.02
        workers: 4
        world_size: 1
        bce_target_thresh: 0.2
    '''

    arguments = read_yaml_string(yaml_string)
    arg_parser = convert_dict_to_argparse(arguments)

    train_parser = argparse.ArgumentParser(
        description="Arguments for training baseline on ImageNet100")
    train_parser.add_argument('--config', default='./config.yaml', type=str, metavar='FILE',
                              help='YAML config file specifying default arguments')
    train_parser.add_argument('--root_dir', default='./', type=str)
    train_parser.add_argument('--epochs', default=100, type=int)
    train_parser.add_argument('--batch_size', default=80, type=int)
    train_parser.add_argument('--image_size', default=160, type=int)
    train_parser.add_argument('--seed', default=42, type=int)
    train_parser.add_argument('--num_classes', default=100, type=int)
    train_parser.add_argument('--num_workers', default=2, type=int)
    train_parser.add_argument('--lr', default=1e-3, type=float)
    train_parser.add_argument('--output_dir', default='./', type=str)
    train_parser.add_argument('--train_time', default=8.8*60*60, type=int)
    parser = argparse.ArgumentParser(
        description="Complete parser")
    args1 = train_parser.parse_args()
    args2 = arg_parser
    args = argparse.Namespace()
    for key, value in vars(args2).items():
        setattr(args, key, value)
    for key, value in vars(args1).items():
        setattr(args, key, value)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    args.prefetcher = False
    
    
    tup = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction((6*(1024)**3)/tup[1], 0)

    EPOCHS = args.epochs
    ACCELARATOR = f'cuda' if torch.cuda.is_available() else 'cpu'

    IMAGE_SIZE = args.img_size
    BATCH_SIZE = args.batch_size

    TRAIN_END_FLAG = -1

    NUM_CLASSES = args.num_classes
    SEED = args.seed
    WARMUP_EPOCHS = args.warmup_epochs
    NUM_WORKERS = args.num_workers
    ROOT_DIR = args.root_dir
    MEAN = IMAGENET_DEFAULT_MEAN
    STD = IMAGENET_DEFAULT_STD
    DECAY_FACTOR = 1
    OUTPUT_DIR = args.output_dir
    MODEL_SAVE_DIR = OUTPUT_DIR

    seed_everything(SEED)

    model1 = Backbone(args)
    model1.to(ACCELARATOR)
    for param in model1.parameters():
        param.requires_grad = True
    data_config = resolve_data_config(
        vars(args), model=model1.encoder, verbose=True)

    print(f"Baseline model:")

    train_loss_fn = BinaryCrossEntropy(
        target_threshold=args.bce_target_thresh).to()
    train_loss_fn = train_loss_fn.to(device=ACCELARATOR)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=ACCELARATOR)

    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(
        args.root_dir, image_size=IMAGE_SIZE, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    train_loader = create_loader(
        train_dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=0,
        interpolation=args.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=None,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        data_type='train'
    )
    validation_loader = create_loader(
        val_dataset,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.num_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        data_type='val'
    )
    test_loader = create_loader(
        test_dataset,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.num_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        data_type='test'
    )

    print(
        f"Length of train loader: {len(train_loader)},Validation loader: {(len(validation_loader))}, Test Loader:{(len(test_loader))}")

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
    mixup_fn = Mixup(**mixup_args)
    steps_per_epoch = len(train_dataset)//(BATCH_SIZE)
    if len(train_dataset) % BATCH_SIZE != 0:
        steps_per_epoch += 1
    optimizer = torch.optim.Adam(model1.parameters(), lr=args.lr)

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, WARMUP_EPOCHS*steps_per_epoch, DECAY_FACTOR*EPOCHS*steps_per_epoch)

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=test_loader,
        device=ACCELARATOR,
        model=model1,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        model_save_dir=MODEL_SAVE_DIR,
        train_criterion=train_loss_fn,
        val_criterion=validate_loss_fn,
        mixup_fn=mixup_fn,
        test_output_dir=OUTPUT_DIR,
        train_time=args.train_time)
    trainer.run(True)