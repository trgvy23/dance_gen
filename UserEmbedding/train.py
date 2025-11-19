from args import parse_train_opt
from dance_gen.UserEmbedding.src.UserEmbedding_with_genre import UserEmbedding


def train(args):
    model = UserEmbedding(args)
    model.train()


if __name__ == "__main__":
    args = parse_train_opt()
    train(args)
