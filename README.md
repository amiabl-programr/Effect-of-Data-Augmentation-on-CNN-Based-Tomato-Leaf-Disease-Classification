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
### Research Design
This study will use a controlled experimental design to investigate the effect of individual image augmentation techniques on CNN-based tomato leaf disease classification.

Four training conditions will be compared:
1. Baseline: No image augmentation.
2. Flip: Horizontal flipping applied to training images.
3. Rotation: Random rotation applied to training images.
4. Brightness: Random brightness adjustment applied to training images.

The baseline provides a reference against which the three augmentation techniques can be evaluated. Each augmentation technique will be tested independently rather than combining multiple transformations. This allows any observed difference in model performance to be attributed more directly to the augmentation condition.
The central independent variable is the augmentation condition, while the CNN architecture, dataset split, preprocessing, optimiser, learning rate, batch size, number of training epochs, and evaluation procedure will remain constant across experiments.

### Dataset
The experiment will use the tomato subset of the PlantVillage dataset. It will include only images belonging to tomato disease/healthy classes.

Before training, the dataset will be inspected to determine:
- the number of images in each class;
- image dimensions and formats;
- class balance;
- the number of classes represented.

The final results will report the class distribution so the classifier's performance can be interpreted in the context of the dataset.
PlantVillage is appropriate for this experiment because it provides substantially more tomato images than smaller field-orientated datasets, allowing the four experimental conditions to be trained and evaluated using a sufficiently large and consistent dataset. However, its controlled imaging environment will be acknowledged as a limitation when discussing generalisation to real agricultural environments.

### Dataset Splitting
The dataset will be divided into:
- Training set: 70%
- Validation set: 15%
- Test set: 15%
The split will be performed before augmentation.

A stratified split will be used so that each disease class maintains approximately the same proportion across the training, validation and test sets.
The test set will remain completely untouched during training. It will only be used for the final evaluation of each experimental condition.
This is important because applying augmentation before splitting the dataset could result in transformed versions of the same original image appearing in different subsets, producing data leakage and overly optimistic performance estimates.
The same split will be reused for all four experiments.

### Image Preprocessing
All images will undergo the same preprocessing regardless of experimental condition.
Images will be:
loaded from the dataset;
resized to a fixed input resolution;
converted to the required image format;
converted to tensors;
normalized using the same normalization parameters.

No augmentation will be applied to validation or test images.
The preprocessing pipeline will therefore be:

Training:
Original image → preprocessing → optional augmentation → normalization → CNN

Validation/Test:
Original image → preprocessing → normalization → CNN

Keeping preprocessing identical across experiments prevents preprocessing differences from becoming an additional experimental variable.


### CNN Model
A small CNN classifier will be used rather than a large state-of-the-art architecture. The purpose of the model is to provide a consistent experimental platform for evaluating augmentation rather than to maximize benchmark accuracy.
The CNN will consist of convolutional layers for feature extraction followed by nonlinear activation functions and pooling layers. The extracted features will then be passed through fully connected layers and a final classification layer.
The final layer will contain one output for each tomato disease class.

## Experiments


## Results


## Analysis


## Limitations


## References
