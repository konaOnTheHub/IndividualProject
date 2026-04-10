"""Eval-time image preprocessing; must match effnet-s_384.ipynb eval_transform."""

from torchvision import transforms

IMG_SIZE = 384
#ImageNet normalization values
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
