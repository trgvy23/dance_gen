from args import parse_train_opt
from src.UserEmbedding import UserEmbedding


def train(args):
    model = UserEmbedding(args.checkpoint)
    model.train(args)


if __name__ == "__main__":
    args = parse_train_opt()
    train(args)
