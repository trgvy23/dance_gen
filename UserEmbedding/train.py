from args import parse_train_opt
from src.UserEmbedding import UserEmbedding


def train(args):
    model = UserEmbedding(args)
    if args.extract_embeddings:
        model.extract_embeddings()
        return
    model.train()


if __name__ == "__main__":
    args = parse_train_opt()
    train(args)
