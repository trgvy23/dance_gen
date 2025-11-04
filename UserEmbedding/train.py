from args import parse_train_opt
from src.UserEmbedding import UserEmbedding

import os

# Make JAX stay on CPU and not preallocate GPU memory
os.environ["JAX_PLATFORMS"] = "cpu"        # new-style
os.environ["JAX_PLATFORM_NAME"] = "cpu"  

def train(args):
    model = UserEmbedding(args.checkpoint)
    model.train(args)


if __name__ == "__main__":
    args = parse_train_opt()
    train(args)
