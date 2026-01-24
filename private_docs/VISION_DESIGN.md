# KladML Vision Architecture Design

This document outlines the architecture for handling Computer Vision tasks, ensuring scalability (Lazy Loading) and flexibility (Siamese/Triplet Networks).

## 1. The Core Problem: Memory vs. Logic
Complex tasks like **Siamese Networks** require specific data pairing logic (Anchor, Positive, Negative) that is distinct from the raw data loading logic.

**Solution: Composable Datasets.**

We split the responsibility into three layers:
1.  **Source Layer (The "What")**: Lazy loading of files.
2.  **Transform Layer (The "How")**: Dynamic augmentation.
3.  **Logic Layer (The "Which")**: Sampling strategy (Classification vs Triplet).

## 2. Architecture Components

### A. Source Layer: `LazyImageSource`
Lightweight wrapper around file paths. Does NOT load images into RAM.
```python
class LazyImageSource:
    def __init__(self, root_dir):
        # Only stores paths: ["path/to/cat.jpg", "path/to/dog.jpg"]
        self.samples = find_images(root_dir) 
        self.class_to_indices = {0: [1, 5, ...], 1: [2, 4, ...]}

    def get_raw(self, index):
        # Reads disk ONLY here
        path = self.samples[index]
        return load_image(path) # Returns PIL/OpenCV image
```

### B. Transform Layer: `AugmentationPipeline`
Receives config (YAML) and builds an Albumentations/Torchvision pipeline.
```python
class AugmentationPipeline:
    def __init__(self, config):
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        return self.transform(image=img)["image"]
```

### C. Logic Layer: Task-Specific Datasets
This is where the "Super Flexibility" lives. The DataModule chooses which Dataset class to instantiate based on the Task.

#### Scenario 1: Standard Classification
Just iterates sequentially.
```python
class ClassificationDataset(Dataset):
    def __init__(self, source, transforms):
        self.source = source
        self.transforms = transforms

    def __getitem__(self, idx):
        img = self.source.get_raw(idx)
        img = self.transforms(img)
        label = self.source.targets[idx]
        return img, label
```

#### Scenario 2: Siamese / Triplet Network
Logic for "Same vs Different".
```python
class TripletDataset(Dataset):
    def __init__(self, source, transforms):
        self.source = source
        self.transforms = transforms

    def __getitem__(self, idx):
        # 1. Anchor: The image at idx
        anchor_img = self.source.get_raw(idx)
        anchor_label = self.source.targets[idx]

        # 2. Positive: Random image of SAME class
        pos_idx = random.choice(self.source.class_to_indices[anchor_label])
        pos_img = self.source.get_raw(pos_idx)

        # 3. Negative: Random image of DIFFERENT class
        neg_label = random.choice([l for l in self.source.classes if l != anchor_label])
        neg_idx = random.choice(self.source.class_to_indices[neg_label])
        neg_img = self.source.get_raw(neg_idx)

        # 4. Transform ALL separately (crucial for robustness)
        return {
            "anchor": self.transforms(anchor_img),
            "positive": self.transforms(pos_img),
            "negative": self.transforms(neg_img)
        }
```

## 3. Integration in `DataModule`

The `VisionDataModule` is the orchestrator.

```python
class VisionDataModule(BaseDataModule):
    def setup(self):
        # 1. Source (Common)
        self.source = LazyImageSource(self.data_dir)
        
        # 2. Transforms (Common)
        self.transforms = AugmentationPipeline(self.config["augmentations"])
        
        # 3. Task Logic (Configurable)
        if self.task_type == "classification":
            self.train_dataset = ClassificationDataset(self.source, self.transforms)
        elif self.task_type == "triplet":
            self.train_dataset = TripletDataset(self.source, self.transforms)
```

## 4. Why this is Future-Proof
1.  **Memory**: `LazyImageSource` handles 1M+ images easily.
2.  **Flexibility**: Adding `QuadrupletLoss` or `ContrastiveLearning` just means adding a new `Dataset` class logic. The loading/augmentation code remains untouched.
3.  **Simplicity**: Users just switch configuration: `task: triplet` vs `task: classification`.
