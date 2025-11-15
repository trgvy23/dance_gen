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
import datetime
import logging

from common.config import JsonConfig

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
from pytorch_metric_learning.utils import accuracy_calculator

from data.dataset import DanceDataset, build_label_mappings
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
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision="bf16")
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

        # DATASETS
        
        print("Building global label mappings...")
        genre2id, dancer2id = build_label_mappings(args.data_path)
        print(f"Num genres:  {len(genre2id)}")
        print(f"Num dancers: {len(dancer2id)}")

        ### LOAD DATASET ###
        print("Loading DanceDataset...")
        train_tensor_dataset_path = os.path.join(
            args.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            args.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            getattr(args, "cache_data", False)
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            self.train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            self.test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:

            self.train_dataset = DanceDataset(
                data_path=args.data_path,
                backup_path=args.processed_data_dir,
                train=True,
                force_reload=getattr(args, "force_reload", False),
                cache_data=getattr(args, "cache_data", False),
                genre2id=genre2id,
                dancer2id=dancer2id,
            )

            self.test_dataset = DanceDataset(
                data_path=args.data_path,
                backup_path=args.processed_data_dir,
                train=False,
                force_reload=getattr(args, "force_reload", False),
                cache_data=getattr(args, "cache_data", False),
                genre2id=genre2id,
                dancer2id=dancer2id,
            )

        self.motionbert = MotionBERTBackbone()
        
        self.motionbert.eval()
        for p in self.motionbert.parameters():
            p.requires_grad_(False)
        
        # TODO: add more arguments for model: hidden size, emb size, etc
        self.user_embedding_net = UserEmbeddingNet(
            self.motionbert,
            num_dancer_class=self.train_dataset.get_dancer_num(),
            num_genre_class=self.train_dataset.get_genre_num(),
        )

        self.optimizer = optim.AdamW(
            self.user_embedding_net.parameters(), weight_decay=0.01
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.hparams.Train.lr_steps, gamma=0.1
        )

        ### LOAD CHECKPOINT IF ANY ###

        if len(self.hparams.Model.checkpoint) > 0:
            print(f"loading weights from {self.hparams.Model.checkpoint}")
            ckp = torch.load(
                os.path.join(self.hparams.Train.log_dir, self.hparams.Model.checkpoint),
                map_location=self.accelerator.device,
            )
            self.user_embedding_net.load_state_dict(ckp["model"], strict=False)
            self.global_step = ckp["step"]
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
        # self.genre_loss_func = losses.TripletMarginLoss(
        #     margin=0.2,
        #     distance=self.distance,
        #     reducer=self.reducer,
        # )

        # self.dancer_loss_func = losses.MultiSimilarityLoss(
        #     alpha=2,
        #     beta=50,
        #     base=0.5,
        #     distance=self.distance,
        #     reducer=self.reducer,
        # )
        # self.genre_loss_func = losses.MultiSimilarityLoss(
        #     alpha=2,
        #     beta=50,
        #     base=0.5,
        #     distance=self.distance,
        #     reducer=self.reducer,
        # )
        self.dancer_loss_func = losses.CrossBatchMemory(
            losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5, distance=self.distance),
            embedding_size=256,
            memory_size=4096,
            miner=miners.MultiSimilarityMiner(epsilon=0.1, distance=self.distance),  # <<—
        )
        self.genre_loss_func = losses.CrossBatchMemory(
            losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5, distance=self.distance),
            embedding_size=256,
            memory_size=4096,
            miner=miners.MultiSimilarityMiner(epsilon=0.1, distance=self.distance),  # <<—
        )

        # NOTE: TripletMarginMiner
        # self.genre_miner = miners.TripletMarginMiner(
        #     margin=0.2,
        #     distance=self.distance,
        #     type_of_triplets="all",
        # )
        # self.dancer_miner = miners.TripletMarginMiner(
        #     margin=0.4,
        #     distance=self.distance,
        #     type_of_triplets="all",
        # )

        # self.dancer_miner = miners.MultiSimilarityMiner(
        #     epsilon=0.1, distance=self.distance
        # )
        # self.genre_miner = miners.MultiSimilarityMiner(
        #     epsilon=0.1, distance=self.distance
        # )

        self.lambda_genre = getattr(args, "lambda_genre", 0.5)
        self.lambda_dancer = getattr(args, "lambda_dancer", 1.0)

        self.use_triplet_reg = getattr(args, "use_triplet_reg", False)
        if self.use_triplet_reg:
            self.triplet_reg = losses.TripletMarginLoss(
                margin=0.2, distance=self.distance
            )
            self.mu_triplet = getattr(args, "mu_triplet", 0.2)

        ### OTHERS ###
        self.log_dir = self.hparams.Train.log_dir
        self.epochs = self.hparams.Train.epochs
        self.batch_size = self.hparams.Train.batch_size

        if args.eval_only:
            self.user_embedding_net = self.accelerator.prepare(self.user_embedding_net)
            self.run_evaluation()
        else:
            self.writer = SummaryWriter(self.log_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            print("Log dir:", self.log_dir)
            self.log_folder_runs = "./runs/{}".format(self.log_dir.split("/")[-1])
            if not os.path.exists(self.log_folder_runs):
                os.system(f"mkdir -p {self.log_folder_runs}")

            # Write configuration file to the log dir
            self.hparams.dump(self.log_dir, "config.json")

            self.print_every = self.hparams.Train.print_every
            self.max_iters = self.hparams.Train.max_iters
            self.save_every = self.hparams.Train.checkpoint_every
            self.eval_every = self.hparams.Train.evaluate_every

    def _extract_embeddings(self, data_loader, is_train_dataloader):
        self.user_embedding_net.eval()
        all_embs = []
        all_dancer_labels = []
        all_genre_labels = []
        
        total_correct_dancer = 0
        total_correct_genre = 0
        total_samples = 0

        with torch.no_grad():
            for video_embedding, video_mask, pose_est, genre_label, dancer_label in data_loader:
                video_embedding = video_embedding.to(self.accelerator.device)
                video_mask = video_mask.to(self.accelerator.device)
                pose_est = pose_est.to(self.accelerator.device)
                genre_label = genre_label.to(self.accelerator.device)
                dancer_label = dancer_label.to(self.accelerator.device)

                with self.accelerator.autocast():
                    if is_train_dataloader:
                        embs, _, _ = self.user_embedding_net(video_embedding, video_mask, pose_est)
                    else:
                        embs, dancer_logits, genre_logits = self.user_embedding_net(video_embedding, video_mask, pose_est)
                    embs = F.normalize(embs, p=2, dim=1)  # for cosine

                all_embs.append(embs.cpu())
                all_dancer_labels.append(dancer_label.cpu())
                all_genre_labels.append(genre_label.cpu())
                
                if not is_train_dataloader:
                    dancer_pred = dancer_logits.argmax(dim=1)
                    genre_pred = genre_logits.argmax(dim=1)

                    total_correct_dancer += (dancer_pred == dancer_label).sum().item()
                    total_correct_genre += (genre_pred == genre_label).sum().item()
                    total_samples += dancer_label.size(0)

        # concat within each process
        all_embs = torch.cat(all_embs, dim=0)
        all_dancer_labels = torch.cat(all_dancer_labels, dim=0)
        all_genre_labels = torch.cat(all_genre_labels, dim=0)

        # gather across processes
        all_embs = self.accelerator.gather_for_metrics(all_embs)
        all_dancer_labels = self.accelerator.gather_for_metrics(all_dancer_labels)
        all_genre_labels = self.accelerator.gather_for_metrics(all_genre_labels)
        
        if is_train_dataloader:
            return all_embs, all_dancer_labels, all_genre_labels
        
        else:
            total_correct_dancer = torch.tensor(
            total_correct_dancer,
            device=self.accelerator.device,
            dtype=torch.long,
            )
            total_correct_genre = torch.tensor(
                total_correct_genre,
                device=self.accelerator.device,
                dtype=torch.long,
            )
            total_samples = torch.tensor(
                total_samples,
                device=self.accelerator.device,
                dtype=torch.long,
            )

            # gather from all processes
            total_correct_dancer_all = self.accelerator.gather_for_metrics(total_correct_dancer)
            total_correct_genre_all = self.accelerator.gather_for_metrics(total_correct_genre)
            total_samples_all = self.accelerator.gather_for_metrics(total_samples)

            dancer_acc = (
                total_correct_dancer_all.sum().float() / total_samples_all.sum().float()
            ).item()
            genre_acc = (
                total_correct_genre_all.sum().float() / total_samples_all.sum().float()
            ).item()
            
            return all_embs, all_dancer_labels, all_genre_labels, dancer_acc, genre_acc

    def run_evaluation(self):

        num_cpus = multiprocessing.cpu_count()

        # make sure datasets already exist (e.g. built in train() or __init__)
        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.Train.batch_size,
            shuffle=False,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=False,
            drop_last=False,
        )
        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.Train.batch_size,
            shuffle=False,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=False,
            drop_last=False,
        )

        # put loaders under Accelerator if you want
        train_data_loader, test_data_loader = self.accelerator.prepare(
            train_data_loader, test_data_loader
        )

        # extract embeddings
        train_embs, train_dancer, train_genre = self._extract_embeddings(
            train_data_loader, is_train_dataloader=True
        )
        test_embs, test_dancer, test_genre, dancer_cls_acc, genre_cls_acc = self._extract_embeddings(test_data_loader, is_train_dataloader=False)

        if not self.accelerator.is_main_process:
            # only main process prints/logs
            self.user_embedding_net.train()
            return

        # to numpy
        train_embs_np = train_embs.cpu().numpy()
        test_embs_np = test_embs.cpu().numpy()
        train_dancer_np = train_dancer.cpu().numpy()
        test_dancer_np = test_dancer.cpu().numpy()
        train_genre_np = train_genre.cpu().numpy()
        test_genre_np = test_genre.cpu().numpy()

        acc_calc = accuracy_calculator.AccuracyCalculator(
            include=("precision_at_1", "r_precision", "mean_average_precision_at_r"),
            device=self.accelerator.device,
        )

        # DANCER RETRIEVAL: query = test, reference = train
        dancer_acc = acc_calc.get_accuracy(
            query=test_embs_np,
            query_labels=test_dancer_np,
            reference=train_embs_np,
            reference_labels=train_dancer_np,
            ref_includes_query=False,
        )

        # GENRE RETRIEVAL: query = test, reference = train
        genre_acc = acc_calc.get_accuracy(
            query=test_embs_np,
            query_labels=test_genre_np,
            reference=train_embs_np,
            reference_labels=train_genre_np,
            ref_includes_query=False,
        )

        print(
            "[Eval] DANCER  | p@1: {:.4f}, R-prec: {:.4f}, mAP@R: {:.4f}".format(
                dancer_acc["precision_at_1"],
                dancer_acc["r_precision"],
                dancer_acc["mean_average_precision_at_r"],
            )
        )
        print(
            "[Eval] GENRE   | p@1: {:.4f}, R-prec: {:.4f}, mAP@R: {:.4f}".format(
                genre_acc["precision_at_1"],
                genre_acc["r_precision"],
                genre_acc["mean_average_precision_at_r"],
            )
        )
        
        print(
            "[Eval] DANCER_CLS | acc: {:.4f}".format(
                dancer_cls_acc
            )
        )
        print(
            "[Eval] GENRE_CLS  | acc: {:.4f}".format(
                genre_cls_acc
            )
        )

        # TensorBoard logging
        if hasattr(self, "writer"):
            self.writer.add_scalar(
                "eval/dancer_p_at_1", dancer_acc["precision_at_1"], self.global_step
            )
            self.writer.add_scalar(
                "eval/dancer_r_precision", dancer_acc["r_precision"], self.global_step
            )
            self.writer.add_scalar(
                "eval/dancer_map_at_r",
                dancer_acc["mean_average_precision_at_r"],
                self.global_step,
            )

            self.writer.add_scalar(
                "eval/genre_p_at_1", genre_acc["precision_at_1"], self.global_step
            )
            self.writer.add_scalar(
                "eval/genre_r_precision", genre_acc["r_precision"], self.global_step
            )
            self.writer.add_scalar(
                "eval/genre_map_at_r",
                genre_acc["mean_average_precision_at_r"],
                self.global_step,
            )
            
            self.writer.add_scalar(
                "eval/test_dancer_cls_acc", dancer_cls_acc, self.global_step
            )
            
            self.writer.add_scalar(
                "eval/test_genre_cls_acc", genre_cls_acc, self.global_step
            )

        self.user_embedding_net.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def log_dict(self, writer, scalars, step, prefix):
        for k, v in scalars.items():
            writer.add_scalar(prefix + "/" + k, v, step)

    def train(self):
        # =========== Prepare Dataloaders ==========
        num_cpus = multiprocessing.cpu_count()

        print("Creating data loaders...")

        all_dancer_labels = torch.tensor(
            [
                self.train_dataset[i][4]
                for i in range(len(self.train_dataset))
            ]
        )
        
        print("Dancer labels:", all_dancer_labels)
        
        #TODO: K ở đây là số sample mỗi class, batch size = P * K (P là số class trong batch)
        K = 8
        sampler = MPerClassSampler(
            labels=all_dancer_labels, m=K, length_before_new_iter=len(all_dancer_labels)
        )
        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.Train.batch_size,
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
            self.scheduler.step()
            avg_dancer_loss = 0.0
            avg_genre_loss = 0.0
            avg_total_loss = 0.0
            
            for batch_idx, (
                video_embedding,
                video_mask,
                pose_est,
                genre_label,
                dancer_label,
            ) in enumerate(load_loop(train_data_loader)):
                video_embedding = video_embedding.to(self.accelerator.device)
                video_mask = video_mask.to(self.accelerator.device)
                pose_est = pose_est.to(self.accelerator.device)
                genre_label = genre_label.to(self.accelerator.device)
                dancer_label = dancer_label.to(self.accelerator.device)

                # TODO: add many arguments and inputs
                with self.accelerator.autocast():
                    embeddings, dancer_logits, genre_logits = self.user_embedding_net(
                        video_embedding, video_mask, pose_est
                    )  # Compute embeddings using the model

                # triplets_d = self.dancer_miner(embeddings, dancer_label)
                # triplets_g = self.genre_miner(embeddings, genre_label)

                # loss_dancer = self.dancer_loss_func(
                #     embeddings, dancer_label, indices_tuple=triplets_d
                # )
                # loss_genre = self.genre_loss_func(
                #     embeddings, genre_label, indices_tuple=triplets_g
                # )
                
                loss_dancer = self.dancer_loss_func(
                    embeddings, dancer_label
                )
                loss_genre = self.genre_loss_func(
                    embeddings, genre_label
                )
                
                # loss = self.lambda_dancer * loss_dancer + self.lambda_genre * loss_genre

                # if self.use_triplet_reg:
                #     T1, T2 = build_hierarchical_triplets(
                #         dancer_label, genre_label
                #     )  # see helper below
                #     L1 = self.triplet_reg(
                #         embeddings, dancer_label, indices_tuple=T1
                #     )  # same-dancer vs same-genre-diff-dancer
                #     L2 = self.triplet_reg(
                #         embeddings, genre_label, indices_tuple=T2
                #     )  # same-genre-diff-dancer vs diff-genre
                #     loss = loss + self.mu_triplet * (L1 + 0.5 * L2)

                # classification losses
                loss_dancer_ce = F.cross_entropy(dancer_logits, dancer_label)
                loss_genre_ce = F.cross_entropy(genre_logits, genre_label)

                # combine (example weights)
                lambda_d_ml = self.lambda_dancer
                lambda_g_ml = self.lambda_genre
                lambda_d_ce = getattr(self, "lambda_d_ce", 1.0)
                lambda_g_ce = getattr(self, "lambda_g_ce", 0.5)

                loss = (
                    lambda_d_ml * loss_dancer
                    + lambda_g_ml * loss_genre
                    + lambda_d_ce * loss_dancer_ce
                    + lambda_g_ce * loss_genre_ce
                )

                # optional hierarchical triplet regularizer
                if self.use_triplet_reg:
                    T1, T2 = build_hierarchical_triplets(dancer_label, genre_label)
                    L1 = self.triplet_reg(embeddings, dancer_label, indices_tuple=T1)
                    L2 = self.triplet_reg(embeddings, genre_label, indices_tuple=T2)
                    loss = loss + self.mu_triplet * (L1 + 0.5 * L2)

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.accelerator.is_main_process:
                    avg_dancer_loss += loss_dancer.item()
                    avg_genre_loss += loss_genre.item()
                    avg_total_loss += loss.item()

                if self.global_step % self.print_every == self.print_every - 1:
                    avg_dancer_loss /= self.print_every
                    avg_genre_loss /= self.print_every
                    avg_total_loss /= self.print_every

                    time = datetime.datetime.now()
                    eta = str(
                        (time - last_time)
                        / self.print_every
                        * (self.max_iters - self.global_step)
                    )
                    last_time = time
                    time = str(time)
                    log_msg = "[{}], eta: {}, iter: {}, progress: {:.2f}%, epoch: {}, dancer loss: {:.4f}, , genre loss: {:.4f}, total loss: {:.4f}".format(
                        time[time.rfind(" ") + 1 : time.rfind(".")],
                        eta[: eta.rfind(".")],
                        self.global_step,
                        (self.global_step / self.max_iters) * 100,
                        i_epoch,
                        avg_dancer_loss,
                        avg_genre_loss,
                        avg_total_loss,
                    )

                    print(log_msg)

                    loss_dict_avg = {
                        "dancer_loss": avg_dancer_loss,
                        "genre_loss": avg_genre_loss,
                        "total_loss": avg_total_loss,
                    }

                    self.log_dict(self.writer, loss_dict_avg, self.global_step, "train")
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )
                    
                    avg_dancer_loss = 0.0
                    avg_genre_loss = 0.0
                    avg_total_loss = 0.0

                # save checkpoint
                if (
                    self.global_step % self.save_every == self.save_every - 1
                    or self.global_step >= self.max_iters
                ):
                    if self.accelerator.is_main_process:
                        save_path = os.path.join(
                            self.log_dir, f"ckp_{self.global_step}.pt"
                        )
                        ckp = {
                            "model": self.accelerator.get_state_dict(
                                self.user_embedding_net
                            ),
                            "optimizer": self.optimizer.state_dict(),
                            "step": self.global_step + 1,
                        }
                        torch.save(ckp, save_path)
                        print(f"Saved checkpoint at step {self.global_step}")

                # Evaluate
                if self.global_step % self.eval_every == self.eval_every - 1 or self.global_step >= self.max_iters:
                    self.run_evaluation()

                    self.writer.add_scalar(
                        "train/epoch",
                        self.global_step / len(train_data_loader),
                        self.global_step,
                    )
                    os.system(f"cp {self.log_dir}/events* {self.log_folder_runs}")
                
                self.global_step += 1
                if self.global_step >= self.max_iters:
                    print("Exit program!")
                    break
                
            if self.global_step >= self.max_iters:
                print("Exit program!")
                break
        os.system(f"cp {self.log_dir}/events* {self.log_folder_runs}")
