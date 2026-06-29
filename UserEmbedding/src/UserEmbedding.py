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

from data.dataset.DanceDataset import DanceDataset
from data.dataset.FreezeUserEmbeddingDanceDataset import FreezeUserEmbeddingDanceDataset
from data.dataset.dataset_utils import build_label_mappings
from src.models import UserEmbeddingNet
from src.backbone import MotionBERTBackbone

from data.smpl_skeleton import SMPLSkeleton
from src.EDGE import DanceDecoder
from src.diffusion import GaussianDiffusion


class MyAccuracyCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_3(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(
            knn_labels,
            query_labels[:, None],
            3,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
        )

    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(
            knn_labels,
            query_labels[:, None],
            5,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
        )

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_3", "precision_at_5"]


class UserEmbedding:
    def __init__(self, args):
        print("Initializing UserEmbedding... Dance Gen Shuffle")
        self.hparams = JsonConfig(args.hparams)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs], mixed_precision="bf16"
        )
        state = AcceleratorState()
        num_processes = state.num_processes

        self.accelerator.wait_for_everyone()
        self.normalizer = None

        # DATASETS

        print("Building global label mappings...")
        genre2id, dancer2id = build_label_mappings(args.data_path)
        print(f"Num genres:  {len(genre2id)}")
        print(f"Num dancers: {len(dancer2id)}")

        ### MODEL ###

        self.motionbert = MotionBERTBackbone()

        ### FROM EDGE ###
        use_baseline_feats = False
        feature_dim = 35 if use_baseline_feats else 4800

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        self.edge = DanceDecoder(
            nfeats=self.repr_dim,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        ).to(self.accelerator.device)

        smpl = SMPLSkeleton(self.accelerator.device)
        self.diffusion = GaussianDiffusion(
            self.edge,
            self.repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        ).to(self.accelerator.device)

        print(
            "Model has {} parameters".format(
                sum(y.numel() for y in self.edge.parameters())
            )
        )

        self.user_embedding_net = UserEmbeddingNet(
            self.motionbert,
            num_dancer_class=len(dancer2id) - 1,
        )

        self.optimizer = optim.AdamW(
            [
                {"params": self.user_embedding_net.parameters(), "lr": 3e-4},
                {"params": self.edge.parameters(), "lr": 3e-4},
            ],
            weight_decay=0.01,
        )

        self.user_embedding_net, self.edge, self.optimizer = self.accelerator.prepare(
            self.user_embedding_net, self.edge, self.optimizer
        )

        self.diffusion.model = self.edge

        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=self.hparams.Train.lr_steps, gamma=0.1
        # )

        ### LOAD CHECKPOINT IF ANY ###

        self.accelerator.wait_for_everyone()

        self.global_step = 0
        self.resume_checkpoint = getattr(args, "checkpoint", None)

        ### OTHERS ###
        self.log_dir = self.hparams.Train.log_dir
        self.epochs = self.hparams.Train.epochs
        self.batch_size = self.hparams.Train.batch_size

        if self.resume_checkpoint:
            print(f"Resuming from checkpoint: {self.resume_checkpoint}")
            ckpt_path = os.path.join(self.log_dir, self.resume_checkpoint)
            self.load_checkpoint(ckpt_path)

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
            self.train_dataset = pickle.load(
                open(train_tensor_dataset_path, "rb"))
            self.test_dataset = pickle.load(
                open(test_tensor_dataset_path, "rb"))
        else:
            self.user_embedding_frozen = getattr(
                args, "user_embedding_frozen", False)
            if self.user_embedding_frozen:
                self.train_dataset = FreezeUserEmbeddingDanceDataset(
                    data_path=args.data_path,
                    backup_path=args.processed_data_dir,
                    train=True,
                    force_reload=getattr(args, "force_reload", False),
                    cache_data=getattr(args, "cache_data", False),
                    genre2id=genre2id,
                    dancer2id=dancer2id,
                )
                self.test_dataset = FreezeUserEmbeddingDanceDataset(
                    data_path=args.data_path,
                    backup_path=args.processed_data_dir,
                    train=False,
                    force_reload=getattr(args, "force_reload", False),
                    cache_data=getattr(args, "cache_data", False),
                    genre2id=genre2id,
                    dancer2id=dancer2id,
                    normalizer=self.train_dataset.normalizer,
                )
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
                    normalizer=self.train_dataset.normalizer,
                )

        if args.eval_only:
            print("Evaluation only mode")
            self.user_embedding_net = self.accelerator.prepare(
                self.user_embedding_net)
            self.run_evaluation(epoch=0, save_vis=True)
            exit(0)
        else:
            self.writer = SummaryWriter(self.log_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            print("Log dir:", self.log_dir)
            self.log_folder_runs = "./runs/{}".format(
                self.log_dir.split("/")[-1])
            if not os.path.exists(self.log_folder_runs):
                os.system(f"mkdir -p {self.log_folder_runs}")

            # Write configuration file to the log dir
            self.hparams.dump(self.log_dir, "config.json")

            self.print_every = self.hparams.Train.print_every
            self.max_iters = self.hparams.Train.max_iters
            self.save_every = self.hparams.Train.checkpoint_every
            self.eval_every = self.hparams.Train.evaluate_every

            # ---- Freezing config ----
            # TODO: Fix this
            self.user_embedding_frozen = getattr(
                args, "user_embedding_frozen", False)
            print("Initial user_embedding_frozen:", self.user_embedding_frozen)
            self.freeze_after_epoch = getattr(args, "freeze_after_epoch", None)
            self.freeze_loss_threshold = getattr(
                args, "freeze_loss_threshold", None)

        self.normalizer = self.test_dataset.normalizer

        # TODO: add more arguments for model: hidden size, emb size, etc

        ############### Metric Learning ###############
        self.distance = distances.CosineSimilarity()
        # self.distance = distances.LpDistance(normalize_embeddings=True, p=2)
        self.reducer = reducers.MeanReducer()

        self.dancer_loss_func = losses.CrossBatchMemory(
            losses.MultiSimilarityLoss(
                alpha=2, beta=50, base=0.5, distance=self.distance
            ),
            embedding_size=256,
            memory_size=4096,
            miner=miners.MultiSimilarityMiner(
                epsilon=0.1, distance=self.distance
            ),  # <<—
        )

        self.lambda_dancer = getattr(args, "lambda_dancer", 1.0)

        self.use_triplet_reg = getattr(args, "use_triplet_reg", False)
        if self.use_triplet_reg:
            print("Using hierarchical triplet regularization")
            self.triplet_reg = losses.TripletMarginLoss(
                margin=0.2, distance=self.distance
            )
            self.mu_triplet = getattr(args, "mu_triplet", 0.2)

    def freeze_user_embedding(self):
        if self.user_embedding_frozen:
            return
        if self.accelerator.is_main_process:
            print("[Freeze] Freezing user_embedding_net parameters")

        # Unwrap in case it's wrapped by DDP/Accelerate
        base_model = self.accelerator.unwrap_model(self.user_embedding_net)
        for p in base_model.parameters():
            p.requires_grad = False

        # Optionally keep it in eval mode to freeze BN stats, etc.
        base_model.eval()

        self.user_embedding_frozen = True

    def save_checkpoint(self, tag=None):
        if not self.accelerator.is_main_process:
            return

        if tag is None:
            ckpt_name = f"ckp_{self.global_step}.pt"
        else:
            ckpt_name = f"{tag}.pt"

        save_path = os.path.join(self.log_dir, ckpt_name)
        checkpoint = {
            "emb_model": self.accelerator.get_state_dict(self.user_embedding_net),
            "edge_model": self.accelerator.get_state_dict(self.edge),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,  # this will be our "start from here" step
            "normalizer": self.normalizer,
        }
        torch.save(checkpoint, save_path)
        print(f"[Checkpoint] Saved to {save_path}")

    def load_checkpoint(self, ckpt_path):
        print(f"[Checkpoint] Loading from {ckpt_path}")
        checkpoint = torch.load(
            ckpt_path, map_location=self.accelerator.device, weights_only=False
        )

        # Models might be wrapped by Accelerate, so unwrap for loading
        self.accelerator.unwrap_model(self.user_embedding_net).load_state_dict(
            checkpoint["emb_model"], strict=False
        )
        self.accelerator.unwrap_model(self.edge).load_state_dict(
            checkpoint["edge_model"], strict=False
        )

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint.get("step", 0)
        self.normalizer = checkpoint.get("normalizer", self.normalizer)

        print(f"[Checkpoint] Loaded step = {self.global_step}")

    def _extract_embeddings(self, data_loader, train=True, epoch=0, save_vis=False):
        self.user_embedding_net.eval()
        self.diffusion.eval()
        if self.user_embedding_frozen:
            return self._extract_embeddings_if_freeze(
                data_loader, train, epoch, save_vis
            )
        all_embs = []
        all_dancer_labels = []

        local_filenames = []

        total_correct_dancer = 0
        total_samples = 0

        with torch.no_grad():
            for (
                video_embedding,
                video_mask,
                pose_est,
                genre_label,
                dancer_label,
                x,
                cond,
                filename,
                wavnames,
            ) in data_loader:
                video_embedding = video_embedding.to(self.accelerator.device)
                video_mask = video_mask.to(self.accelerator.device)
                pose_est = pose_est.to(self.accelerator.device)
                dancer_label = dancer_label.to(self.accelerator.device)

                # with self.accelerator.autocast():
                embs, dancer_logits, _ = self.user_embedding_net(
                    video_embedding, video_mask, pose_est
                )

                if train is False:
                    # generate a sample

                    print("Generating Sample")

                    render_count, horizon, _ = cond.shape
                    render_count = 2
                    horizon = 486
                    print("Render count:", render_count, "Horizon:", horizon)
                    shape = (render_count, horizon, self.repr_dim)

                    cond = cond.to(self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        embs[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join("render", "train"),
                        fk_out=os.path.join(
                            "render", f"{epoch}", "motion_result"),
                        name=wavnames[:render_count],
                        sound=True,
                    )

                all_embs.append(embs.cpu())
                all_dancer_labels.append(dancer_label.cpu())

                local_filenames.extend(list(filename))

                dancer_pred = dancer_logits.argmax(dim=1)

                total_correct_dancer += (dancer_pred ==
                                         dancer_label).sum().item()
                total_samples += dancer_label.size(0)

        # concat within each process
        local_embs = torch.cat(all_embs, dim=0)  # [N_local, D]
        local_labels = torch.cat(all_dancer_labels, 0)  # [N_local]

        # concat within each process
        all_embs = torch.cat(all_embs, dim=0)
        all_dancer_labels = torch.cat(all_dancer_labels, dim=0)

        # gather across processes
        all_embs = self.accelerator.gather_for_metrics(all_embs)
        all_dancer_labels = self.accelerator.gather_for_metrics(
            all_dancer_labels)

        total_correct_dancer = torch.tensor(
            total_correct_dancer,
            device=self.accelerator.device,
            dtype=torch.long,
        )
        total_samples = torch.tensor(
            total_samples,
            device=self.accelerator.device,
            dtype=torch.long,
        )

        # gather from all processes
        total_correct_dancer_all = self.accelerator.gather_for_metrics(
            total_correct_dancer
        )
        total_samples_all = self.accelerator.gather_for_metrics(total_samples)

        dancer_acc = (
            total_correct_dancer_all.sum().float() / total_samples_all.sum().float()
        ).item()

        if save_vis:
            split = "train" if train else "test"
            # all_filenames_nested = self.accelerator.gather_object(local_filenames)

            if save_vis and self.accelerator.is_main_process:
                # flatten nested list: [[rank0_files...], [rank1_files...], ...] -> [all_files...]
                # all_filenames = [f for sublist in local_filenames for f in sublist]
                print(len(local_filenames), "filenames collected for viz")

                # sanity check ordering/size
                assert local_embs.shape[0] == len(
                    local_filenames
                ), f"Mismatch: embeddings {local_embs.shape[0]} vs filenames {len(local_filenames)}"

                save_name = f"embeddings_{split}.npz"
                save_path = os.path.join(self.log_dir, save_name)

                np.savez(
                    save_path,
                    embeddings=local_embs.cpu().numpy(),  # [N_total, D]
                    dancer_labels=local_labels.cpu().numpy(),  # [N_total]
                    # [N_total] of strings
                    filenames=np.array(local_filenames),
                )
                print(
                    f"[Eval] (main) Saved {split} embeddings for viz to {save_path}")

        return all_embs, all_dancer_labels, dancer_acc

    def _extract_embeddings_if_freeze(
        self, data_loader, train=True, epoch=0, save_vis=False
    ):
        all_embs = []
        all_dancer_labels = []

        local_filenames = []

        total_correct_dancer = 0
        total_samples = 0

        with torch.no_grad():
            for (
                embs,
                dancer_logits,
                _,
                pose_est,
                genre_label,
                dancer_label,
                x,
                cond,
                filename,
                wavnames,
            ) in data_loader:
                pose_est = pose_est.to(self.accelerator.device)
                dancer_label = dancer_label.to(self.accelerator.device)

                if train is False:
                    # generate a sample

                    print("Generating Sample")

                    render_count, horizon, _ = cond.shape
                    render_count = 2
                    horizon = 486
                    print("Render count:", render_count, "Horizon:", horizon)
                    shape = (render_count, horizon, self.repr_dim)

                    cond = cond.to(self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        embs[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join("render", "train"),
                        fk_out=os.path.join(
                            "render", f"{epoch}", "motion_result"),
                        name=wavnames[:render_count],
                        sound=True,
                    )

                all_embs.append(embs.cpu())
                all_dancer_labels.append(dancer_label.cpu())

                local_filenames.extend(list(filename))

                dancer_pred = dancer_logits.argmax(dim=1)

                total_correct_dancer += (dancer_pred ==
                                         dancer_label).sum().item()
                total_samples += dancer_label.size(0)

        # concat within each process
        local_embs = torch.cat(all_embs, dim=0)  # [N_local, D]
        local_labels = torch.cat(all_dancer_labels, 0)  # [N_local]

        # concat within each process
        all_embs = torch.cat(all_embs, dim=0)
        all_dancer_labels = torch.cat(all_dancer_labels, dim=0)

        # gather across processes
        all_embs = self.accelerator.gather_for_metrics(all_embs)
        all_dancer_labels = self.accelerator.gather_for_metrics(
            all_dancer_labels)

        total_correct_dancer = torch.tensor(
            total_correct_dancer,
            device=self.accelerator.device,
            dtype=torch.long,
        )
        total_samples = torch.tensor(
            total_samples,
            device=self.accelerator.device,
            dtype=torch.long,
        )

        # gather from all processes
        total_correct_dancer_all = self.accelerator.gather_for_metrics(
            total_correct_dancer
        )
        total_samples_all = self.accelerator.gather_for_metrics(total_samples)

        dancer_acc = (
            total_correct_dancer_all.sum().float() / total_samples_all.sum().float()
        ).item()

        if save_vis:
            split = "train" if train else "test"
            # all_filenames_nested = self.accelerator.gather_object(local_filenames)

            if save_vis and self.accelerator.is_main_process:
                # flatten nested list: [[rank0_files...], [rank1_files...], ...] -> [all_files...]
                # all_filenames = [f for sublist in local_filenames for f in sublist]
                print(len(local_filenames), "filenames collected for viz")

                # sanity check ordering/size
                assert local_embs.shape[0] == len(
                    local_filenames
                ), f"Mismatch: embeddings {local_embs.shape[0]} vs filenames {len(local_filenames)}"

                save_name = f"embeddings_{split}.npz"
                save_path = os.path.join(self.log_dir, save_name)

                np.savez(
                    save_path,
                    embeddings=local_embs.cpu().numpy(),  # [N_total, D]
                    dancer_labels=local_labels.cpu().numpy(),  # [N_total]
                    # [N_total] of strings
                    filenames=np.array(local_filenames),
                )
                print(
                    f"[Eval] (main) Saved {split} embeddings for viz to {save_path}")

        return all_embs, all_dancer_labels, dancer_acc

    def run_evaluation(self, epoch, save_vis=False):

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
        (
            train_embs,
            train_dancer,
            dancer_cls_train_acc,
        ) = self._extract_embeddings(
            train_data_loader, train=True, epoch=0, save_vis=save_vis
        )
        test_embs, test_dancer, dancer_cls_acc = self._extract_embeddings(
            test_data_loader, train=False, epoch=epoch, save_vis=save_vis
        )

        # to numpy
        train_embs_np = train_embs.cpu().numpy()
        test_embs_np = test_embs.cpu().numpy()
        train_dancer_np = train_dancer.cpu().numpy()
        test_dancer_np = test_dancer.cpu().numpy()

        # acc_calc = accuracy_calculator.AccuracyCalculator(
        #     include=("precision_at_1", "r_precision", "mean_average_precision_at_r"),
        #     device=self.accelerator.device,
        # )

        acc_calc = MyAccuracyCalculator(
            include=(
                "precision_at_1",
                "precision_at_3",
                "precision_at_5",
                "r_precision",
                "mean_average_precision_at_r",
            ),
            device=self.accelerator.device,
        )

        print("Query:", set(test_dancer_np.tolist()))
        print("Reference:", set(train_dancer_np.tolist()))
        print(
            "Intersection:",
            set(test_dancer_np.tolist()) & set(train_dancer_np.tolist()),
        )

        # DANCER RETRIEVAL: query = test, reference = train
        dancer_acc = acc_calc.get_accuracy(
            query=test_embs_np,
            query_labels=test_dancer_np,
            reference=train_embs_np,
            reference_labels=train_dancer_np,
            ref_includes_query=False,
        )

        # print(
        #     "[Eval] DANCER  | p@1: {:.4f}, R-prec: {:.4f}, mAP@R: {:.4f}".format(
        #         dancer_acc["precision_at_1"],
        #         dancer_acc["r_precision"],
        #         dancer_acc["mean_average_precision_at_r"],
        #     )
        # )

        print(
            "[Eval] DANCER  | "
            "p@1: {:.4f}, p@3: {:.4f}, p@5: {:.4f}, "
            "R-prec: {:.4f}, mAP@R: {:.4f}".format(
                dancer_acc["precision_at_1"],
                dancer_acc["precision_at_3"],
                dancer_acc["precision_at_5"],
                dancer_acc["r_precision"],
                dancer_acc["mean_average_precision_at_r"],
            )
        )

        print("[Eval] DANCER_CLS | acc: {:.4f}".format(dancer_cls_acc))

        print("[Train] DANCER_CLS | acc: {:.4f}".format(dancer_cls_train_acc))

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
                "eval/test_dancer_cls_acc", dancer_cls_acc, self.global_step
            )

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def log_dict(self, writer, scalars, step, prefix):
        for k, v in scalars.items():
            writer.add_scalar(prefix + "/" + k, v, step)

    def train(self):
        if self.user_embedding_frozen:
            self.train_with_freeze_user_embedding()
            return
        # =========== Prepare Dataloaders ==========
        num_cpus = multiprocessing.cpu_count()

        print("Creating data loaders...")

        all_dancer_labels = torch.tensor(
            [self.train_dataset[i][4] for i in range(len(self.train_dataset))]
        )

        print("new run")

        # TODO: K ở đây là số sample mỗi class, batch size = P * K (P là số class trong batch)
        K = 8
        sampler = MPerClassSampler(
            labels=all_dancer_labels, m=K, length_before_new_iter=len(
                all_dancer_labels)
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

        train_data_loader = self.accelerator.prepare(train_data_loader)

        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )

        s_epoch = int(self.global_step / len(train_data_loader))

        self.user_embedding_net.train()

        last_time = datetime.datetime.now()
        self.accelerator.wait_for_everyone()
        for i_epoch in range(s_epoch, self.hparams.Train.epochs):

            avg_dancer_loss = 0.0
            avg_dancer_cls_loss = 0.0
            avg_total_loss = 0.0
            avg_pose_recon_loss = 0.0
            avg_edge_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0

            self.user_embedding_net.train()
            self.diffusion.train()

            for batch_idx, (
                video_embedding,
                video_mask,
                pose_est,
                genre_label,
                dancer_label,
                x,
                cond,
                filename,
                wavnames,
            ) in enumerate(load_loop(train_data_loader)):
                video_embedding = video_embedding.to(self.accelerator.device)
                video_mask = video_mask.to(self.accelerator.device)
                pose_est = pose_est.to(self.accelerator.device)
                dancer_label = dancer_label.to(self.accelerator.device)

                # TODO: add many arguments and inputs
                # with self.accelerator.autocast():
                #     embeddings, dancer_logits, pose_recon = self.user_embedding_net(
                #         video_embedding, video_mask, pose_est
                #     )  # Compute embeddings using the model
                if self.user_embedding_frozen:
                    with torch.no_grad():
                        embeddings, dancer_logits, pose_recon = self.user_embedding_net(
                            video_embedding, video_mask, pose_est
                        )
                else:
                    with self.accelerator.autocast():
                        embeddings, dancer_logits, pose_recon = self.user_embedding_net(
                            video_embedding, video_mask, pose_est
                        )

                loss_dancer = self.dancer_loss_func(embeddings, dancer_label)

                # classification losses
                loss_dancer_ce = F.cross_entropy(dancer_logits, dancer_label)

                # combine (example weights)
                lambda_d_ml = self.lambda_dancer
                lambda_d_ce = getattr(self, "lambda_d_ce", 1.0)

                # pose_est, pose_recon: [B, T, 17, 3]
                # print(pose_recon.shape, pose_est.shape)
                loss_pose_recon = F.mse_loss(pose_recon, pose_est)

                lambda_d_ml = self.lambda_dancer
                lambda_d_ce = getattr(self, "lambda_d_ce", 1.0)
                lambda_recon = getattr(self, "lambda_recon", 1.0)

                total_loss, (edge_loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                    x, cond, embeddings, t_override=None
                )

                if self.user_embedding_frozen:
                    # Phase 2: only train EDGE / diffusion
                    loss = total_loss
                else:
                    # Phase 1: full joint training
                    loss = (
                        lambda_d_ml * loss_dancer
                        + lambda_d_ce * loss_dancer_ce
                        + lambda_recon * loss_pose_recon
                        + total_loss
                    )

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.accelerator.is_main_process:
                    avg_dancer_loss += loss_dancer.item()
                    avg_dancer_cls_loss += loss_dancer_ce.item()
                    avg_pose_recon_loss += loss_pose_recon.item()
                    avg_total_loss += loss.item()

                    avg_edge_loss += edge_loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()

                self.global_step += 1

            # self.scheduler.step()

            if i_epoch % self.print_every == self.print_every - 1:

                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.user_embedding_net.eval()
                    avg_dancer_loss /= self.print_every
                    avg_dancer_cls_loss /= self.print_every
                    avg_total_loss /= self.print_every
                    avg_pose_recon_loss /= self.print_every

                    avg_edge_loss /= self.print_every
                    avg_vloss /= self.print_every
                    avg_fkloss /= self.print_every
                    avg_footloss /= self.print_every

                    time = datetime.datetime.now()
                    eta = str(
                        (time - last_time)
                        / self.print_every
                        * (self.max_iters - self.global_step)
                    )
                    last_time = time
                    time = str(time)
                    log_msg = "[{}], eta: {}, iter: {}, progress: {:.2f}%, epoch: {}, dancer loss: {:.4f}, \
                        avg_dancer_cls loss: {:.4f}, avg_pose_recon: {:.4f}, total loss: {:.4f}\
                            avg_edge_loss: {:.4f}, avg_vloss: {:.4f}, avg_fkloss: {:.4f}, avg_footloss: {:.4f}".format(
                        time[time.rfind(" ") + 1: time.rfind(".")],
                        eta[: eta.rfind(".")],
                        self.global_step,
                        (self.global_step / self.max_iters) * 100,
                        i_epoch,
                        avg_dancer_loss,
                        avg_dancer_cls_loss,
                        avg_pose_recon_loss,
                        avg_total_loss,
                        avg_edge_loss,
                        avg_vloss,
                        avg_fkloss,
                        avg_footloss,
                    )

                    print(log_msg)

                    loss_dict_avg = {
                        "dancer_loss": avg_dancer_loss,
                        "avg_dancer_cls_loss": avg_dancer_cls_loss,
                        "avg_pose_recon_loss": avg_pose_recon_loss,
                        "total_loss": avg_total_loss,
                        "avg_edge_loss": avg_edge_loss,
                        "avg_vloss": avg_vloss,
                        "avg_fkloss": avg_fkloss,
                        "avg_footloss": avg_footloss,
                    }

                    self.log_dict(self.writer, loss_dict_avg,
                                  self.global_step, "train")
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )

                    if not self.user_embedding_frozen:
                        cond_epoch = (
                            self.freeze_after_epoch is not None
                            and i_epoch >= self.freeze_after_epoch
                        )
                        cond_loss = (
                            self.freeze_loss_threshold is not None
                            and avg_total_loss > 0
                            and avg_total_loss < self.freeze_loss_threshold
                        )
                        if cond_epoch or cond_loss:
                            print("Freezing user_embedding_net as per schedule...")
                            self.freeze_user_embedding()

                    avg_dancer_loss = 0.0
                    avg_dancer_cls_loss = 0.0
                    avg_pose_recon_loss = 0.0
                    avg_total_loss = 0.0

                    avg_edge_loss = 0.0
                    avg_vloss = 0.0
                    avg_fkloss = 0.0
                    avg_footloss = 0.0

            # save checkpoint
            if (
                i_epoch % self.save_every == self.save_every - 1
                or self.global_step >= self.max_iters
            ):
                if self.accelerator.is_main_process:
                    save_path = os.path.join(
                        self.log_dir, f"ckp_{self.global_step}.pt")
                    ckp = {
                        "emb_model": self.accelerator.get_state_dict(
                            self.user_embedding_net
                        ),
                        "edge_model": self.accelerator.get_state_dict(self.edge),
                        "optimizer": self.optimizer.state_dict(),
                        "step": self.global_step + 1,
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckp, save_path)
                    # draw a music from the test dataset

                    print(f"Saved checkpoint at step {self.global_step}")

            # Evaluate
            if (
                i_epoch % self.eval_every == self.eval_every - 1
                or self.global_step >= self.max_iters
            ):
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.user_embedding_net.eval()
                    self.diffusion.eval()
                    self.run_evaluation(i_epoch)

                    self.writer.add_scalar(
                        "train/epoch",
                        self.global_step / len(train_data_loader),
                        self.global_step,
                    )
                    os.system(
                        f"cp {self.log_dir}/events* {self.log_folder_runs}")

            if self.global_step >= self.max_iters:
                print("Exit program!")
                break
        os.system(f"cp {self.log_dir}/events* {self.log_folder_runs}")

    def train_with_freeze_user_embedding(self):
        # =========== Prepare Dataloaders ==========
        num_cpus = multiprocessing.cpu_count()

        print("Creating data loaders...")

        all_dancer_labels = torch.tensor(
            [self.train_dataset[i][5] for i in range(len(self.train_dataset))]
        )

        print("new run")

        # TODO: K ở đây là số sample mỗi class, batch size = P * K (P là số class trong batch)
        K = 8
        sampler = MPerClassSampler(
            labels=all_dancer_labels, m=K, length_before_new_iter=len(
                all_dancer_labels)
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

        train_data_loader = self.accelerator.prepare(train_data_loader)

        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )

        s_epoch = int(self.global_step / len(train_data_loader))

        self.user_embedding_net.train()

        last_time = datetime.datetime.now()
        self.accelerator.wait_for_everyone()
        print("Starting training with USER EMBEDDING FROZEN")
        for i_epoch in range(s_epoch, self.hparams.Train.epochs):

            avg_dancer_loss = 0.0
            avg_dancer_cls_loss = 0.0
            avg_total_loss = 0.0
            avg_pose_recon_loss = 0.0
            avg_edge_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0

            self.user_embedding_net.eval()
            self.diffusion.train()

            for batch_idx, (
                embs,
                dancer_logits,
                pose_recon,
                pose_est,
                genre_label,
                dancer_label,
                x,
                cond,
                filename,
                wavnames,
            ) in enumerate(load_loop(train_data_loader)):
                dancer_label = dancer_label.to(self.accelerator.device)

                loss_dancer = self.dancer_loss_func(embs, dancer_label)

                # classification losses
                loss_dancer_ce = F.cross_entropy(dancer_logits, dancer_label)

                # combine (example weights)
                lambda_d_ml = self.lambda_dancer
                lambda_d_ce = getattr(self, "lambda_d_ce", 1.0)

                # pose_est, pose_recon: [B, T, 17, 3]
                # print(pose_recon.shape, pose_est.shape)
                loss_pose_recon = F.mse_loss(pose_recon, pose_est)

                lambda_d_ml = self.lambda_dancer
                lambda_d_ce = getattr(self, "lambda_d_ce", 1.0)
                lambda_recon = getattr(self, "lambda_recon", 1.0)

                total_loss, (edge_loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                    x, cond, embs, t_override=None
                )

                if self.user_embedding_frozen:
                    # Phase 2: only train EDGE / diffusion
                    loss = total_loss
                else:
                    # Phase 1: full joint training
                    loss = (
                        lambda_d_ml * loss_dancer
                        + lambda_d_ce * loss_dancer_ce
                        + lambda_recon * loss_pose_recon
                        + total_loss
                    )

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.accelerator.is_main_process:
                    avg_dancer_loss += loss_dancer.item()
                    avg_dancer_cls_loss += loss_dancer_ce.item()
                    avg_pose_recon_loss += loss_pose_recon.item()
                    avg_total_loss += loss.item()

                    avg_edge_loss += edge_loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()

                self.global_step += 1

            # self.scheduler.step()

            if i_epoch % self.print_every == self.print_every - 1:

                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.user_embedding_net.eval()
                    avg_dancer_loss /= self.print_every
                    avg_dancer_cls_loss /= self.print_every
                    avg_total_loss /= self.print_every
                    avg_pose_recon_loss /= self.print_every

                    avg_edge_loss /= self.print_every
                    avg_vloss /= self.print_every
                    avg_fkloss /= self.print_every
                    avg_footloss /= self.print_every

                    time = datetime.datetime.now()
                    eta = str(
                        (time - last_time)
                        / self.print_every
                        * (self.max_iters - self.global_step)
                    )
                    last_time = time
                    time = str(time)
                    log_msg = "[{}], eta: {}, iter: {}, progress: {:.2f}%, epoch: {}, dancer loss: {:.4f}, \
                        avg_dancer_cls loss: {:.4f}, avg_pose_recon: {:.4f}, total loss: {:.4f}\
                            avg_edge_loss: {:.4f}, avg_vloss: {:.4f}, avg_fkloss: {:.4f}, avg_footloss: {:.4f}".format(
                        time[time.rfind(" ") + 1: time.rfind(".")],
                        eta[: eta.rfind(".")],
                        self.global_step,
                        (self.global_step / self.max_iters) * 100,
                        i_epoch,
                        avg_dancer_loss,
                        avg_dancer_cls_loss,
                        avg_pose_recon_loss,
                        avg_total_loss,
                        avg_edge_loss,
                        avg_vloss,
                        avg_fkloss,
                        avg_footloss,
                    )

                    print(log_msg)

                    loss_dict_avg = {
                        "dancer_loss": avg_dancer_loss,
                        "avg_dancer_cls_loss": avg_dancer_cls_loss,
                        "avg_pose_recon_loss": avg_pose_recon_loss,
                        "total_loss": avg_total_loss,
                        "avg_edge_loss": avg_edge_loss,
                        "avg_vloss": avg_vloss,
                        "avg_fkloss": avg_fkloss,
                        "avg_footloss": avg_footloss,
                    }

                    self.log_dict(self.writer, loss_dict_avg,
                                  self.global_step, "train")
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )

                    if not self.user_embedding_frozen:
                        cond_epoch = (
                            self.freeze_after_epoch is not None
                            and i_epoch >= self.freeze_after_epoch
                        )
                        cond_loss = (
                            self.freeze_loss_threshold is not None
                            and avg_total_loss > 0
                            and avg_total_loss < self.freeze_loss_threshold
                        )
                        if cond_epoch or cond_loss:
                            print("Freezing user_embedding_net as per schedule...")
                            self.freeze_user_embedding()

                    avg_dancer_loss = 0.0
                    avg_dancer_cls_loss = 0.0
                    avg_pose_recon_loss = 0.0
                    avg_total_loss = 0.0

                    avg_edge_loss = 0.0
                    avg_vloss = 0.0
                    avg_fkloss = 0.0
                    avg_footloss = 0.0

            # save checkpoint
            if (
                i_epoch % self.save_every == self.save_every - 1
                or self.global_step >= self.max_iters
            ):
                if self.accelerator.is_main_process:
                    save_path = os.path.join(
                        self.log_dir, f"ckp_{self.global_step}.pt")
                    ckp = {
                        "emb_model": self.accelerator.get_state_dict(
                            self.user_embedding_net
                        ),
                        "edge_model": self.accelerator.get_state_dict(self.edge),
                        "optimizer": self.optimizer.state_dict(),
                        "step": self.global_step + 1,
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckp, save_path)
                    # draw a music from the test dataset

                    print(f"Saved checkpoint at step {self.global_step}")

            # Evaluate
            if (
                i_epoch % self.eval_every == self.eval_every - 1
                or self.global_step >= self.max_iters
            ):
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.user_embedding_net.eval()
                    self.diffusion.eval()
                    self.run_evaluation(i_epoch)

                    self.writer.add_scalar(
                        "train/epoch",
                        self.global_step / len(train_data_loader),
                        self.global_step,
                    )
                    os.system(
                        f"cp {self.log_dir}/events* {self.log_folder_runs}")

            if self.global_step >= self.max_iters:
                print("Exit program!")
                break
        os.system(f"cp {self.log_dir}/events* {self.log_folder_runs}")

    def extract_embeddings(self, num_workers=4):
        self.user_embedding_net.eval()
        self.diffusion.eval()

        data_path = self.train_dataset.data_path

        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.Train.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.Train.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        train_data_loader, test_data_loader = self.accelerator.prepare(
            train_data_loader, test_data_loader
        )

        with torch.no_grad():
            for is_train, data_loader in zip(
                [True, False], [train_data_loader, test_data_loader]
            ):
                data_dir = os.path.join(
                    data_path, "train" if is_train else "test")
                assert os.path.exists(
                    data_dir
                ), f"Data directory {data_dir} does not exist"
                pbar = tqdm(
                    data_loader,
                    desc=f"Extracting embeddings from {'train' if is_train else 'test'} set",
                )

                for (
                    video_embedding,
                    video_mask,
                    pose_est,
                    _,
                    _,
                    _,
                    _,
                    filename,
                    _,
                ) in pbar:
                    video_embedding = video_embedding.to(
                        self.accelerator.device)
                    video_mask = video_mask.to(self.accelerator.device)
                    pose_est = pose_est.to(self.accelerator.device)

                    embs, dance_logits, pose_recon = self.user_embedding_net(
                        video_embedding, video_mask, pose_est
                    )

                    base_dir = os.path.join(data_dir, "feat_embeddings")
                    os.makedirs(base_dir, exist_ok=True)

                    embs = embs.cpu().numpy()
                    dance_logits = dance_logits.cpu().numpy()
                    pose_recon = pose_recon.cpu().numpy()

                    for fname_path, emb, logits, pose in zip(
                        filename, embs, dance_logits, pose_recon
                    ):
                        fname = os.path.splitext(
                            os.path.basename(fname_path))[0]

                        pbar.set_postfix({"current_file": fname})

                        save_path = os.path.join(base_dir, f"{fname}.npz")
                        assert save_path.endswith(
                            ".npz"
                        ), f"Expected .npz extension, got {save_path}"
                        tqdm.write(f"Saving embedding at {save_path}")

                        np.savez(
                            save_path, embs=emb, dance_logits=logits, pose_recon=pose
                        )
