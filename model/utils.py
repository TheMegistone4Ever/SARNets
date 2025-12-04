import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./dataset/bright_dataset"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./test_results"

BATCH_SIZE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
INPUT_SIZE = 372
NUM_CLASSES = 4
NUM_WORKERS = 4
PIN_MEMORY = True
EARLY_STOPPING_PATIENCE = 15

COLOR_MAP = {
    0: [50, 200, 50],  # No Damage - зелений
    1: [255, 255, 0],  # Minor - жовтий
    2: [255, 128, 0],  # Major - помаранчевий
    3: [255, 0, 0]  # Destroyed - червоний
}

DAMAGE_LABELS = ["No Damage", "Minor", "Major", "Destroyed"]


def to_tensor(shape):
    return torch.tensor(shape)


def crop_tensor(tensor, target_tensor):
    """
    Crop tensor to match target_tensor size (center crop).

    Args:
        tensor: tensor to crop (B, C, H, W) or (B, H, W)
        target_tensor: reference tensor

    Returns:
        cropped tensor
    """
    target_size = to_tensor(target_tensor.shape[-2:])
    start = (to_tensor(tensor.shape[-2:]) - target_size) // 2
    return tensor[tuple(slice(None) for _ in range(tensor.ndim - 2))
                  + tuple(slice(int(s), int(e)) for s, e in zip(start.tolist(), (target_size + start).tolist()))]
