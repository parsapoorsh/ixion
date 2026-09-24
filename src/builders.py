"""Transform factories for training and evaluation."""

from torchvision import transforms as T

import constants


def build_train_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    erasing_p: float = 0.25,
    crop_policy: str = "random",
) -> T.Compose:
    """Label-safe augmentation only. No spatial flip/rotate (would corrupt labels)."""
    if crop_policy == "none":
        spatial = [T.Resize(constants.INPUT_SIZE)]
    elif crop_policy == "random":
        spatial = [T.Resize(constants.RESIZE_SIZE), T.RandomCrop(constants.INPUT_SIZE)]
    elif crop_policy == "center":
        spatial = [T.Resize(constants.RESIZE_SIZE), T.CenterCrop(constants.INPUT_SIZE)]
    else:
        raise ValueError(f"unknown crop policy: {crop_policy}")

    transforms = [
        *spatial,
        T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.05,
        ),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
    if erasing_p > 0:
        # After normalize so erased patches are mean-gray in model space.
        transforms.append(
            T.RandomErasing(
                p=erasing_p,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0,
            )
        )
    return T.Compose(transforms)


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
