from kaggle_datasets import KaggleDatasets

from .train import train_model
from .utils import load_device_strategy


DEVICE = "TPU"
AUTO, strategy, tpu = load_device_strategy(DEVICE)
REPLICAS = strategy.num_replicas_in_sync
print("Num replicas:", REPLICAS)

SEED = 0
N_FOLDS = 5
IMG_SIZE = 384  # Available size: [128, 256, 384, 512, 767]
BATCH_SIZE = 32
EPOCHS = 12
TTA_ROUNDS = 10


if __name__ == "__main__":
    GCS_PATH = KaggleDatasets().get_gcs_path(f'melanoma-{IMG_SIZE}x{IMG_SIZE}')
    train_model(GCS_PATH, IMG_SIZE, N_FOLDS, BATCH_SIZE, EPOCHS, TTA_ROUNDS,
                DEVICE, tpu, strategy, REPLICAS, random_state=SEED)
