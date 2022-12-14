# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms

from optim import LARS
from continual_learner import ContinualLearner
import utils
from embeddingreg import embedding
from sampleconfig import config
from configClasses import DefaultConfig, Embedding
# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--ewc_lambda', default=1., type=float, metavar='EWC_L',
                    help='regularization strenght')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--dist_train', default=False, type=bool,
                    metavar='DD', help='choose if distributed training')
parser.add_argument('--ER_task_count', default=0, type=int,
                    metavar='ER_task_count', help='write which continual task it is')
parser.add_argument('--diz', default='./checkpoint/diz.npy', type=Path,
                    metavar='DIZ', help='dict with embeddings previous task infos')

args = parser.parse_args()
ewc = False # set this to false to stop EWC
er = True if args.ER_task_count > 0 else False
allowed_classes = None

def main():
    args.ngpus_per_node = torch.cuda.device_count()
    if args.dist_train:
      if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
      else: 
        print("SLURM not in Environment")
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
        args.ER_task_count = args.ER_task_count
    else:
        print("Single node dist training")
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
        args.ER_task_count = args.ER_task_count
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    ### set the config
    config = DefaultConfig()
    config.EPOCHS = 10
    config.IS_INCREMENTAL = True
    config.LR = 1e-1
    # config.BATCH_SIZE = 64
    # config.EWC_IMPORTANCE = 0.5
    # config.EWC_SAMPLE_SIZE = 100
    # config.OPTIMIZER = 'Adam'
    config.CL_TEC = embedding
    config.USE_CL = True

    config.NEXT_TASK_LR = None
    config.NEXT_TASK_EPOCHS = None

    # writer = SummaryWriter()

    
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)

    # if ewc:
    if er:
      model.ER_task_count = args.ER_task_count
      print('ER task count: ', model.ER_task_count)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)


    dataset = torchvision.datasets.DatasetFolder(
      root=args.data,
      loader=npy_loader,
      extensions=('.npy', '.jpg', '.tiff', '.png', '.tif'),
      transform = Transform())

    # dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size,
        pin_memory=True, sampler=sampler, drop_last=True)

    cont_learn_tec = embedding(model, loader, config, gpu, per_device_batch_size)
    

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        print('Loading checkpoints...')
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth', map_location='cpu')
        if args.ER_task_count > 0:
            start_epoch = 0
            # with open(args.checkpoint_dir / args.diz, 'r') as JSON:
            #     diz = json.load(JSON)
            diz = np.load(args.checkpoint_dir / args.diz, allow_pickle = True)
            cont_learn_tec.get_info(diz)
            
            print('ER info correctly loaded!')
        else:
            start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Checkpoints loaded!')
    else:
        print('No checkpoints found')
        start_epoch = 0

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)

            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                bt_loss = model.forward(y1, y2)
                # if ewc:
                #   loss, bt_loss, ewc_loss = loss
                
                # L1 regularize the  barlow twins loss
                if config.L1_REG > 0:
                    l1_loss = 0.0
                    for name, param in model.named_parameters():
                        l1_loss += torch.sum(abs(param))
                    bt_loss = bt_loss + config.L1_REG * l1_loss
            scaler.scale(bt_loss).backward(retain_graph=True)

            if cont_learn_tec is not None:
                # embedding regularizer -- compute penalty and add it.
                _, er_loss = cont_learn_tec(current_task=args.ER_task_count, batch=(y1, y2))
                if er: 
                    loss = bt_loss + er_loss
                    # print(f"Epoch: {epoch} | Loss with ER: {loss.detach()} | penalty: {er_loss}")
                    # writer.add_scalar("Loss with ER", loss.detach(), step)
                    scaler.scale(loss).backward(retain_graph=True)
                else:
                    loss = bt_loss
            
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    # if ewc:
                    if er:
                        stats['ER_loss'] = er_loss
                        stats['BT_loss'] = bt_loss.item()
                        stats['ER_task_count'] = model.module.ER_task_count
                        
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
            # print("Checkpoints salvati")
            
    final_epoch = epoch if epoch else start_epoch
    
    if args.rank == 0:
        # save final model
        # if ewc:
        if er:
            diz = {}
            embeddings, embeddings_images = cont_learn_tec.get_embeddings()
            diz['embeddings'], diz['embeddings_images'] = embeddings, embeddings_images
            # with open(args.checkpoint_dir / f'er_info{args.ER_task_count+1}.json', 'w') as fp:
            #      json.dump(diz, fp)
            np.save(args.checkpoint_dir / f'er_info{args.ER_task_count+1}.npy', diz)
            print('ER task count: ', model.module.ER_task_count)
            print('Embeddings computed')
        state = dict(epoch=final_epoch, model=model.state_dict(), optimizer=optimizer.state_dict())
        torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
        torch.save(model.module.backbone.state_dict(),args.checkpoint_dir / 'resnet50_task{}.pth'.format(args.ER_task_count))
        # writer.close()

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass

def npy_loader(path):
  # sample = torch.from_numpy(np.load(path))
  if path.endswith('.npy'):
    sample = np.load(path)
    # print(path)
    if 'Potsdam' in path:
        sample = np.transpose(sample, (1,2,0))
    # print("shape: ", sample.shape)
    sample = sample.astype(np.uint8)
    sample = Image.fromarray(sample)
  else:
    sample = Image.open(path)
  return sample #torch.from_numpy(sample)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(ContinualLearner):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.b_s = self.args.batch_size

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag

        if ewc:
            # Add EWC-loss
            ewc_loss = self.ewc_loss()
            # print("EWC-loss: ", ewc_loss.item(), '\tBT-loss:', loss.item())
            tot_loss = loss + self.ewc_lambda * ewc_loss
            
            return [tot_loss, loss, ewc_loss]
        
        else:
            return loss
        
    def embedding(self, y1, y2):
        """embedding for the ER."""
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        z1 = z1.view(z1.size(0), -1)
        z2 = z2.view(z2.size(0), -1)
        return z1, z2
        

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    
if __name__ == '__main__':
    main()
