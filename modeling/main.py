import argparse
import yaml

from kaggle_datasets import KaggleDatasets

from .train import train_model
from .utils import load_device_strategy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('device', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.device not in ("CPU", "GPU", "TPU"):
        raise ValueError("Invalid device. Available: CPU, GPU, TPU. Got: %s" % \
                         args.device)

    # Useful for training on Kaggle
    auto, strategy, tpu = load_device_strategy(args.device)
    replicas = strategy.num_replicas_in_sync
    print("Num remplicas:", replicas)

    dataset_path = KaggleDatasets().get_gcs_path(
        f'melanoma-{cfg["input"]["image_size"]}x{cfg["input"]["image_size"]}')
    train_model(dataset_path, cfg, args.device, tpu, strategy, replicas)
