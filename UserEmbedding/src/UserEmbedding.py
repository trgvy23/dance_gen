import multiprocessing
import os
import numpy as np
import argparse
import errno
import math
import pickle
from tqdm import tqdm
from time import time
from functools import partial
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import DanceDataset
from src.backbones import DSTformer

class UserEmbedding:
    # TODO: build input pipeline for MotionBERT
    def __init__(self, args, checkpoint_path=""):
        # self.motionbert_backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep,
        #                         depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                         maxlen=args.maxlen, num_joints=args.num_joints)

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device, weights_only=False
            )

    def train(self, args):
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )

        train_dataset = DanceDataset(
            data_path=args.data_path,
            backup_path=args.processed_data_dir,
            train=True,
            force_reload=args.force_reload,
            no_cache=args.no_cache,
        )

        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )

        for video, pose_est in load_loop(train_data_loader):
            print(video.shape, pose_est.shape)
            break
