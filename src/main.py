import argparse
import yaml

from train import train_model
from utils import load_device_strategy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--data_path', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.device not in ("CPU", "GPU", "TPU"):
        raise ValueError("Invalid device. Available: CPU, GPU, TPU. Got: %s" % \
                         args.device)

    auto, strategy, tpu = load_device_strategy(args.device)
    replicas = strategy.num_replicas_in_sync
    print("Num remplicas:", replicas)

    train_model(args.data_path, cfg, auto, replicas, strategy)
