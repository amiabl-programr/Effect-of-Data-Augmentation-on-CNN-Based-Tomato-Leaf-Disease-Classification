"""
Dataset handling for tomato-subset PlantVillage classification.

Expects data/ to be laid out as an ImageFolder:

    data/
      Tomato___Bacterial_spot/
        img1.jpg ...
      Tomato___Early_blight/
        img1.jpg ...
      ...

Handles:
  - class distribution inspection
  - stratified train/val/test split (BEFORE augmentation, per section 3.3)
  - condition-specific augmentation (baseline / flip / rotation / brightness)
  - identical preprocessing across all conditions (section 3.4)
"""
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 128
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def inspect_dataset(root: str):
    """Prints class counts / balance. Run this once before splitting."""
    ds = ImageFolder(root)
    counts = Counter([label for _, label in ds.samples])
    print(f"Classes found: {len(ds.classes)}")
    for idx, cls in enumerate(ds.classes):
        print(f"  {cls}: {counts.get(idx, 0)} images")
    return ds.classes, counts


def build_augmentation(condition: str, params: dict):
    """
    Returns the training-only augmentation transform for a given condition.
    Validation/test never receive this.
    """
    if condition == "baseline":
        return transforms.Compose([])  

    if condition == "flip":
        p = params.get("flip_prob", 0.5)
        return transforms.RandomHorizontalFlip(p=p)

    if condition == "rotation":
        degrees = params.get("rotation_degrees", 20)
        return transforms.RandomRotation(degrees=degrees)

    if condition == "brightness":
        factor = params.get("brightness_factor", 0.3)
        return transforms.ColorJitter(brightness=factor)

    raise ValueError(f"Unknown augmentation condition: {condition}")


def get_base_transform():
    """Preprocessing shared by ALL conditions and ALL splits (section 3.4)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


class TransformSubset(Dataset):
    """Wraps a subset of (path, label) samples with a chosen transform pipeline."""
    def __init__(self, samples, loader, transform):
        self.samples = samples
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        img = self.transform(img)
        return img, label


def stratified_split(root: str, seed: int, train_frac=0.70, val_frac=0.15):
    """
    Splits BEFORE augmentation to avoid leakage.
    Returns raw (path, label) sample lists for train/val/test, plus class list.
    """
    ds = ImageFolder(root)
    samples = ds.samples  
    labels = [label for _, label in samples]

    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples, labels, train_size=train_frac, stratify=labels, random_state=seed
    )
    val_size_of_temp = val_frac / (1 - train_frac)
    val_samples, test_samples, _, _ = train_test_split(
        temp_samples, temp_labels, train_size=val_size_of_temp,
        stratify=temp_labels, random_state=seed
    )
    return train_samples, val_samples, test_samples, ds.classes, ds.loader


def build_dataloaders(root: str, condition: str, seed: int, batch_size: int,
                       aug_params: dict, num_workers: int = 2):
    """Builds train/val/test DataLoaders for a given augmentation condition."""
    train_samples, val_samples, test_samples, classes, loader = stratified_split(root, seed)

    base_tf = get_base_transform()
    aug_tf = build_augmentation(condition, aug_params)

    train_transform = transforms.Compose([aug_tf, base_tf])
    eval_transform = base_tf  

    train_ds = TransformSubset(train_samples, loader, train_transform)
    val_ds = TransformSubset(val_samples, loader, eval_transform)
    test_ds = TransformSubset(test_samples, loader, eval_transform)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes


def compute_class_weights(train_samples, num_classes):
    """w_c = N / (C * N_c), per section. Only use if inspection reveals imbalance."""
    counts = Counter([label for _, label in train_samples])
    n_total = len(train_samples)
    weights = [n_total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)
