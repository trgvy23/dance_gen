from args import parse_train_opt
from src.UserEmbedding import UserEmbedding

import os

# Make JAX stay on CPU and not preallocate GPU memory
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.0")


def train(args):
    model = UserEmbedding(args.checkpoint)
    model.train(args)


if __name__ == "__main__":
    args = parse_train_opt()
    train(args)
