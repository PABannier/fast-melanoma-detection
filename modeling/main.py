import tensorflow as tf
from .preprocessing import transform
from .utils import load_device_strategy


AUTO, REPLICAS = load_device_strategy("TPU")
print("Num replicas:", REPLICAS)
