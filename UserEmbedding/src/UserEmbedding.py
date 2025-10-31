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

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.samplers import MPerClassSampler

from data.dataset import DanceDataset
from src.models import UserEmbeddingNet


def build_hierarchical_triplets(y_d, y_g):
    # y_d, y_g: [B] int tensors
    B = y_d.size(0)
    device = y_d.device
    eye = torch.eye(B, dtype=torch.bool, device=device)

    same_d = (y_d[:, None] == y_d[None, :]) & ~eye
    same_g = (y_g[:, None] == y_g[None, :]) & ~eye
    diff_g = ~same_g & ~eye

    # A: pos = same dancer, neg = same genre & different dancer
    A_pos = torch.where(same_d)
    A_negs_mask = same_g & ~same_d
    a1, p1, n1 = [], [], []
    for a, p in zip(*A_pos):
        negs = torch.where(A_negs_mask[a])[0]
        if len(negs) == 0:
            continue
        a1 += [a.item()] * len(negs)
        p1 += [p.item()] * len(negs)
        n1 += negs.tolist()

    # B: pos = same genre diff dancer, neg = different genre
    B_pos_mask = same_g & ~same_d
    a2, p2, n2 = [], [], []
    for a, p in zip(*torch.where(B_pos_mask)):
        negs = torch.where(diff_g[a])[0]
        if len(negs) == 0:
            continue
        a2 += [a.item()] * len(negs)
        p2 += [p.item()] * len(negs)
        n2 += negs.tolist()

    to_t = lambda xs: torch.tensor(xs, device=device, dtype=torch.long)
    return (to_t(a1), to_t(p1), to_t(n1)), (to_t(a2), to_t(p2), to_t(n2))


class UserEmbedding:
    def __init__(self, args, checkpoint_path=""):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.accelerator.device,
                weights_only=False,
            )

        self.user_embedding_net = UserEmbeddingNet()

        self.optimizer = optim.Adam(
            self.user_embedding_net.parameters(), lr=0.0005, weight_decay=0.01
        )
        # self.num_epochs = 1

        ############### Metric Learning ###############
        self.distance = distances.CosineSimilarity()
        # self.distance = distances.LpDistance(normalize_embeddings=True, p=2)
        self.reducer = reducers.MeanReducer()

        # NOTE: TripletMarginLoss has been commented out and replaced with MultiSimilarityLoss
        # self.dancer_loss_func = losses.TripletMarginLoss(
        #     margin=0.4,
        #     distance=self.distance,
        #     reducer=self.reducer,
        # )
        # self.gerne_loss_func = losses.TripletMarginLoss(
        #     margin=0.2,
        #     distance=self.distance,
        #     reducer=self.reducer,
        # )

        self.dancer_loss_func = losses.MultiSimilarityLoss(
            alpha=2,
            beta=50,
            base=0.5,
            distance=self.distance,
            reducer=self.reducer,
        )
        self.genre_loss_func = losses.MultiSimilarityLoss(
            alpha=2,
            beta=50,
            base=0.5,
            distance=self.distance,
            reducer=self.reducer,
        )

        # NOTE: TripletMarginMiner
        # self.gerne_miner = miners.TripletMarginMiner(
        #     margin=0.2,
        #     distance=self.distance,
        #     type_of_triplets="all",
        # )
        # self.dancer_miner = miners.TripletMarginMiner(
        #     margin=0.4,
        #     distance=self.distance,
        #     type_of_triplets="all",
        # )

        self.dancer_miner = miners.MultiSimilarityMiner(
            epsilon=0.1, distance=self.distance
        )
        self.genre_miner = miners.MultiSimilarityMiner(
            epsilon=0.1, distance=self.distance
        )

        self.lambda_genre = getattr(args, "lambda_genre", 0.5)
        self.lambda_dancer = getattr(args, "lambda_dancer", 1.0)

        self.use_triplet_reg = getattr(args, "use_triplet_reg", False)
        if self.use_triplet_reg:
            self.triplet_reg = losses.TripletMarginLoss(
                margin=0.2, distance=self.distance
            )
            self.mu_triplet = getattr(args, "mu_triplet", 0.15)

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train(self, args):
        print("Loading DanceDataset...")
        train_dataset = DanceDataset(
            data_path=args.data_path,
            backup_path=args.processed_data_dir,
            train=True,
            force_reload=getattr(args, "force_reload", False),
            cache_data=getattr(args, "cache_data", False),
        )

        num_cpus = multiprocessing.cpu_count()

        print("Creating data loaders...")

        all_dancer_labels = torch.tensor(
            [
                train_dataset[i][3]  # index 3 = dancer_label in your tuple
                for i in range(len(train_dataset))
            ]
        )

        sampler = MPerClassSampler(
            labels=all_dancer_labels, m=4, length_before_new_iter=len(all_dancer_labels)
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=False,
            drop_last=True,
        )

        # train_data_loader = DataLoader(
        #     train_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=min(int(num_cpus * 0.75), 32),
        #     pin_memory=True,
        #     drop_last=True,
        # )

        self.user_embedding_net, self.optimizer, train_data_loader = (
            self.accelerator.prepare(
                self.user_embedding_net, self.optimizer, train_data_loader
            )
        )

        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )

        self.accelerator.wait_for_everyone()

        for batch_idx, (video, pose_est, gerne_label, dancer_label) in enumerate(
            load_loop(train_data_loader)
        ):
            video = video.to(self.accelerator.device)
            pose_est = pose_est.to(self.accelerator.device)
            gerne_label = gerne_label.to(self.accelerator.device)
            dancer_label = dancer_label.to(self.accelerator.device)

            # TODO: forward pass
            # TODO: add many arguments and inputs
            with self.accelerator.autocast():
                embeddings = self.user_embedding_net(
                    video, pose_est
                )  # Compute embeddings using the model

            triplets_d = self.dancer_miner(embeddings, dancer_label)
            triplets_g = self.gerne_miner(embeddings, gerne_label)

            loss_dancer = self.dancer_loss_func(
                embeddings, dancer_label, indices_tuple=triplets_d
            )
            loss_gerne = self.genre_loss_func(
                embeddings, gerne_label, indices_tuple=triplets_g
            )
            loss = self.lambda_dancer * loss_dancer + self.lambda_genre * loss_gerne

            if self.use_triplet_reg:
                T1, T2 = build_hierarchical_triplets(
                    dancer_label, gerne_label
                )  # see helper below
                L1 = self.triplet_reg(
                    embeddings, dancer_label, indices_tuple=T1
                )  # same-dancer vs same-genre-diff-dancer
                L2 = self.triplet_reg(
                    embeddings, gerne_label, indices_tuple=T2
                )  # same-genre-diff-dancer vs diff-genre
                loss = loss + self.mu_triplet * (L1 + 0.5 * L2)

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.accelerator.is_main_process and (batch_idx % 2 == 0):
                print(
                    f"[{batch_idx}] L_dancer={loss_dancer.item():.4f}  L_genre={loss_gerne.item():.4f}  total={loss.item():.4f}"
                )
