import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()
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

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
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
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    ### Pretrains ###
    parser.add_argument(
        "--motionbert_pretrain",
        type=str,
        default="checkpoint/motionbert/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        help="pretrained MotionBERT checkpoint path",
    )
    opt = parser.parse_args()
    return opt
