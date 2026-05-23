"""Transform factories for training and evaluation."""

from torchvision import transforms as T

import constants


def build_train_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    """Photometric-only augmentation. Any spatial flip/rotate would corrupt the label."""
    return T.Compose([
        T.Resize(constants.RESIZE_SIZE),
        T.RandomCrop(constants.INPUT_SIZE),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


eval_preprocess = T.Compose([
    T.Resize(constants.RESIZE_SIZE),
    T.CenterCrop(constants.INPUT_SIZE),
    T.ToTensor(),
])


def build_eval_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    return T.Compose([
        *eval_preprocess.transforms,
        T.Normalize(mean, std),
    ])
