import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hparams",
        default="/raid/ltnghia02/vyttt/dance_gen_v2/configs/user_embedding.json",
        type=str,
        help="hyper parameters config file path",
    )
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument(
        "--exp_name", default="user_embedding", help="save to project/name"
    )
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )

    # parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval_only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--cache_data", action="store_true", help="cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="trained checkpoint path (optional)",
    )

    # freeze schedule
    parser.add_argument(
        "--freeze_after_epoch",
        type=int,
        default=500,
        help=(
            "Epoch after which to freeze user_embedding_net and only train EDGE. "
            "Set None (default) to disable."
        ),
    )
    parser.add_argument(
        "--freeze_loss_threshold",
        type=float,
        default=0.1,
        help=(
            "Freeze user_embedding_net once average total loss < this value. "
            "Set None (default) to disable this condition."
        ),
    )
    parser.add_argument(
        "--user_embedding_frozen",
        type=bool,
        default=False,
        help=(
            "Whether to freeze user_embedding_net and only train EDGE. "
        ),
    )
    ### Pretrains ###
    parser.add_argument(
        "--motionbert_pretrain",
        type=str,
        default="checkpoint/motionbert/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        help="pretrained MotionBERT checkpoint path",
    )
    parser.add_argument(
        "--use_triplet_reg",
        action="store_true",
        default=False,
        help="use triplet regularization",
    )
    opt = parser.parse_args()
    return opt
