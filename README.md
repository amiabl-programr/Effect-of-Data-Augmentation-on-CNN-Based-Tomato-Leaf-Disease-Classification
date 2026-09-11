# Effect of Data Augmentation on CNN-Based Tomato Leaf Disease Classification

## Research Question
To what extent does data augmentation affect the generalisation performance of a CNN when classifying tomato leaf diseases?

## Hypothesis
Appropriate data augmentation will improve generalization and reduce overfitting, while excessively strong augmentation may reduce classification performance.

## Literature review
Tomato diseases can significantly reduce crop productivity, making early, accurate disease identification challenging in agricultural applications. In recent years, deep learning, particularly convolutional neural networks (CNNs), has become a widely used approach for plant disease classification and detection. A review of tomato disease and pest detection research by Jelali et al. (2024) found that CNN-based approaches remain dominant, with architectures including VGG, ResNet, DenseNet, R-CNN and custom CNN models being frequently used. More recently, transformer-based and CNN-transformer hybrid approaches have also emerged as alternatives for improving feature representation and robustness.

A major dataset that is used in tomato plant disease research is the PlantVillage dataset. It contains more than 54,000 laboratory-captured images covering multiple plant species and disease categories. Its controlled imaging conditions and relatively uniform backgrounds make it useful for developing and comparing classification models. However, these same characteristics limit its representation of real agricultural environments. Real-world images may contain varying illumination, shadows, overlapping leaves, complex backgrounds, and occlusions, which are largely absent from PlantVillage. Consequently, models that achieve very high performance on PlantVillage may not generalise as well to field conditions.
PlantDoc was introduced as a more realistic alternative containing images captured under less controlled conditions. Although such datasets better represent practical agricultural environments, they are considerably smaller and can contain substantial class imbalance. For a small experimental study, PlantVillage therefore provides a larger, more consistent dataset for controlled investigation, but its limitations must be considered when interpreting the results.

During training, data augmentation is a common technique used to increase the effective diversity of training data by applying transformations to existing images. Shorten and Khoshgoftaar (2019) categorise augmentation broadly into data-warping approaches and methods that generate or combine additional samples. Traditional image-space transformations include geometric operations such as rotation, flipping, cropping and scaling, as well as colour and intensity transformations. More advanced approaches include MixUp, CutMix and generative methods such as GAN-based augmentation.
The primary motivation for augmentation is to improve generalisation and reduce overfitting. By exposing a model to different versions of training images, augmentation can encourage it to learn features that remain useful under variations in viewpoint, illumination and other image conditions. It is particularly useful when the available dataset is limited or imbalanced. Augmentation can therefore act as a form of regularisation by reducing a model's tendency to memorise individual training examples.

However, augmentation does not automatically improve performance. Its effectiveness depends on whether the transformations suit the underlying task and dataset. Transformations that introduce unrealistic variations can distort important visual features or cause the augmented training distribution to differ from the evaluation distribution. Xu et al. (2023) similarly emphasise selecting augmentation methods based on the dataset and task characteristics. Traditional transformations are attractive because they are simple and computationally inexpensive, but their label-preserving assumptions limit how aggressively they can be applied.

Wagle et al. (2021) provide an example of data augmentation, investigating tomato disease classification using ResNet architectures and PlantVillage images. Their experiments compared the original data with geometrically augmented datasets and reported improvements in aggregate classification accuracy. However, their expanded augmentation condition also substantially increased the dataset size, making it difficult to attribute all performance improvements specifically to the transformations themselves. Their class-level results also indicate that improvements were not necessarily uniform across all disease classes.
Joshi et al. (2025) provide a more detailed investigation by testing individual augmentation operations, including flipping, rotation, saturation, Gaussian blur and noise. Their results suggest that different augmentation operations can produce different effects, with rotation producing the greatest improvement among the individually tested techniques. However, their final proposed augmentation pipeline combined several transformations and also changed other experimental factors, such as training duration and dataset size. This makes it difficult to determine each component's independent contribution to the final performance.

Some of the studies reviewed above establish that data augmentation can improve the performance and generalisation of deep learning models for plant and tomato disease recognition. Nevertheless, augmentation is frequently introduced as a bundled preprocessing strategy rather than being systematically compared at the level of individual transformations. Several studies combine operations such as flipping, rotation, brightness adjustment, noise and other transformations, making it difficult to determine whether particular augmentation techniques are more beneficial than others.
This creates an opportunity for a controlled experiment in which individual augmentation techniques are evaluated under otherwise identical training conditions. Rather than introducing a new CNN architecture, the proposed study focuses on isolating the effect of augmentation itself. A CNN will therefore be trained under four conditions: no augmentation (baseline), flipping, rotation, and brightness adjustment. The same dataset split, model architecture and training configuration will be maintained across conditions.

In addition to overall classification performance, examining class-level results can show whether augmentation's effect is consistent across tomato disease categories. This matters because improved aggregate accuracy does not necessarily mean every disease class benefits equally. The experiment will therefore investigate not only whether augmentation changes CNN classification performance but also whether the type of augmentation influences performance distribution across disease classes.
The study is deliberately limited to the PlantVillage tomato subset to provide enough data for a controlled, reproducible experiment. Results will be interpreted within the limitations of PlantVillage, particularly its controlled imaging conditions and limited representation of real-world field environments.



## Method

### Research design
A controlled experiment compares four training conditions — **baseline** (no augmentation), **flip**, **rotation**, and **brightness** — each trained independently rather than combined, so any performance difference can be attributed to the augmentation condition itself. Augmentation is the only variable that changes between runs; CNN architecture, dataset split, preprocessing, optimiser, learning rate, batch size, epoch count and evaluation procedure are held constant.

### Dataset
The tomato-disease subset of PlantVillage, laid out as an `ImageFolder` (`data/<class_name>/*.jpg`). `src/inspect_dataset.py` reports the number of classes, per-class image counts and class balance before any training happens.

### Splitting
`stratified_split()` in `src/dataset.py` performs a stratified 70/15/15 train/val/test split (via `sklearn.train_test_split`) **before** any augmentation is applied, so transformed copies of the same source image can't leak across splits. The split is seeded (`seed=42`) and reused identically across all four conditions; the test set is never touched until final evaluation.

### Preprocessing
Every image — regardless of condition or split — is resized to 128×128, converted to a tensor, and normalized with ImageNet mean/std. Augmentation, where applicable, is inserted only into the training pipeline, ahead of this shared preprocessing:

- **Train:** image → augmentation (if any) → resize/normalize → CNN
- **Val/Test:** image → resize/normalize → CNN

### Augmentation conditions
Implemented in `build_augmentation()` (`src/dataset.py`), parameters fixed per condition in `configs/*.yaml`:

| Condition | Transform | Parameter |
|---|---|---|
| Baseline | none (identity) | — |
| Flip | `RandomHorizontalFlip` | p = 0.5 |
| Rotation | `RandomRotation` | ±20° |
| Brightness | `ColorJitter(brightness=...)` | factor = 0.3 |

### Model
`SmallCNN` (`src/model.py`): three convolutional blocks (32→64→128 channels, 3×3 kernels, ReLU, 2×2 max-pool), flattened into an FC(256) layer with dropout (0.3), then a final FC layer with one output per class. Logits are trained with softmax cross-entropy; the network is intentionally small since the goal is an isolated augmentation comparison, not benchmark accuracy.

### Training
Adam optimiser, lr = 0.001, batch size 32, 20 epochs, categorical cross-entropy loss (`src/train.py`). Class weighting (`w_c = N / (C·N_c)`) is available via `compute_class_weights()` and currently enabled (`use_class_weights: true`) in all four configs. The checkpoint with the best validation accuracy is saved per run to `results/checkpoints/`.

### Reproducibility & tracking
`set_seed()` (`src/utils.py`) fixes the seed across Python, NumPy, PyTorch and cuDNN (deterministic mode). Every run logs to Weights & Biases under the `tomato-leaf-augmentation` project, grouped by condition (`baseline` / `flip` / `rotation` / `brightness`) so runs can be compared without manual record-keeping.


### Evaluation
`src/evaluate.py` loads a checkpoint, runs inference on the held-out test split, and reports accuracy, macro precision/recall/F1, a full per-class classification report, and a confusion matrix heatmap saved to `results/confusion_matrices/`.

## Experiments


## Results


## Analysis


## Limitations


## References
