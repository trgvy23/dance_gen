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
import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.samplers import MPerClassSampler

from data.dataset import DanceDataset
from src.models import UserEmbeddingNet
from src.backbone import MotionBERTBackbone


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
    def __init__(self, args):
        self.hparams = JsonConfig(args.hparams)
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        self.accelerator.wait_for_everyone()

        # checkpoint = None
        # if checkpoint_path != "":
        #     checkpoint = torch.load(
        #         checkpoint_path,
        #         map_location=self.accelerator.device,
        #         weights_only=False,
        #     )

        self.motionbert = MotionBERTBackbone()
        #TODO: add more arguments for model: hidden size, emb size, etc
        self.user_embedding_net = UserEmbeddingNet(self.motionbert)

        self.optimizer = optim.AdamW(
            self.user_embedding_net.parameters(), weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.hparams.Train.lr_steps, gamma=0.1)
        
        ### LOAD CHECKPOINT IF ANY ###

        if len(self.hparams.Model.checkpoint) > 0:
            print(f"loading weights from {self.hparams.Model.checkpoint}")
            ckp = torch.load(join(self.hparams.Train.log_dir, self.hparams.Model.checkpoint), map_location=self.accelerator.device)
            self.user_embedding_net.load_state_dict(ckp['model'], strict=False)
            self.global_step = ckp['step']
        else:
            self.global_step = 0
            
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
            
        ### OTHERS ###
        self.log_dir = self.hparams.Train.log_dir
        self.epochs = getattr(args, "epochs", 2000)
        
        if args.eval_only:
            self.run_evaluation()
        else:
            self.writer = SummaryWriter(self.log_dir)
            if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print("Log dir:", self.log_dir)
            log_folder_runs = "./runs/{}".format(self.log_dir.split('/')[-1])
            if not os.path.exists(log_folder_runs):
                os.system(f"mkdir -p {log_folder_runs}")
                
            # Write configuration file to the log dir
            self.hparams.dump(log_dir, 'config.json')
            
            self.print_every = self.hparams.Train.print_every
            self.max_iters = self.hparams.Train.max_iters
            self.save_every = self.hparams.Train.checkpoint_every
            self.eval_every = self.hparams.Train.evaluate_every

    def run_evaluation(self):
        pass
    
    def prepare(self, objects):
        return self.accelerator.prepare(*objects)
    
    def log_dict(self, writer, scalars, step, prefix):
        for k, v in scalars.items():
            writer.add_scalar(prefix + "/" + k, v, step)

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
        
        s_epoch = int(self.global_step / len(train_data_loader))
        
        last_time = datetime.datetime.now()
        for i_epoch in range(s_epoch, self.hparams.Train.epochs):
            self.optimizer.step()
            for batch_idx, (video_embedding, pose_est, gerne_label, dancer_label) in enumerate(
                load_loop(train_data_loader)
            ):
                video_embedding = video_embedding.to(self.accelerator.device)
                pose_est = pose_est.to(self.accelerator.device)
                gerne_label = gerne_label.to(self.accelerator.device)
                dancer_label = dancer_label.to(self.accelerator.device)

                # TODO: add many arguments and inputs
                with self.accelerator.autocast():
                    embeddings = self.user_embedding_net(
                        video_embedding, pose_est
                    )  # Compute embeddings using the model

                triplets_d = self.dancer_miner(embeddings, dancer_label)
                triplets_g = self.genre_miner(embeddings, gerne_label)

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
                
                if self.accelerator.is_main_process:
                    avg_dancer_loss += loss_dancer.item()
                    avg_genre_loss += loss_gerne.item()
                    avg_total_loss += loss.item()
                    
                if self.global_step % self.print_every == self.print_every - 1:
                    avg_dancer_loss /= self.print_every
                    avg_genre_loss /= self.print_every
                    avg_total_loss /= self.print_every
                    
                    time = datetime.datetime.now()
                    eta = str((time - last_time) / self.print_every * (self.max_iters - self.global_step))
                    last_time = time
                    time = str(time)
                    log_msg = "[{}], eta: {}, iter: {}, progress: {:.2f}%, epoch: {}, dancer loss: {:.4f}, , gerne loss: {:.4f}, total loss: {:.4f}".format(
                        time[time.rfind(' ') + 1:time.rfind('.')],
                        eta[:eta.rfind('.')],
                        self.global_step,
                        (self.global_step / self.max_iters) * 100,
                        i_epoch,
                        avg_dancer_loss,
                        avg_genre_loss,
                        avg_total_loss,
                    )
                    
                    print(log_msg)
                    
                    loss_dict_avg = {
                        'dancer_loss': avg_dancer_loss,
                        'genre_loss': avg_genre_loss,
                        'total_loss': avg_total_loss,
                    }
                    
                    self.log_dict(self.writer, loss_dict_avg, self.global_step, 'train')
                    self.writer.add_scalar('train/lr',
                                    self.optimizer.param_groups[0]["lr"],
                                    self.global_step)
                    
                # save checkpoint
                if self.global_step % self.save_every == self.save_every - 1 and self.global_step < self.max_iters:
                    if self.accelerator.is_main_process:
                        save_path = os.path.join(log_dir, f"ckp_{global_step}.pt")
                        ckp = {
                            'model': self.accelerator.get_state_dict(self.user_embedding_net),
                            'optimizer': self.optimizer.state_dict(),
                            'step': self.global_step + 1,
                        }
                        torch.save(ckp, save_path)
                        print(f"Saved checkpoint at step {self.global_step}")
                        
                # Evaluate
                if self.global_step % self.eval_every == self.eval_every - 1:
                    run_evaluation()

                    writer.add_scalar('train/epoch',
                                    self.global_step / len(train_data_loader),
                                    self.global_step)
                    os.system(f"cp {log_dir}/events* {log_folder_runs}")

                
                self.global_step += 1
                if self.global_step >= self.max_iters:
                    print("Exit program!")
                    break
