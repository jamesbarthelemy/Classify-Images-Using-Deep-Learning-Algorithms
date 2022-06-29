# Classify images using deep learning algorithms

Most computer vision algorithms use a convolution neural network, or CNN. 
Like basic feedforward neural networks, CNNs learn from inputs, adjusting their parameters to make a prediction. 
However, what makes CNNs special is their ability to extract features from images.

In this project which aims to classify dog images according to the dog's breed, I will first implement my own CNN inspired by the famous InceptionV1 model. 
Next, I'll demonstrate how transfer learning outperforms this baseline using other popular pre-trained models. 

# Table of contents

1. [Problem description](#problem-description)
2. [Data analysis overview](#data-analysis-overview)
3. [Features engineering](#features-engineering)
4. [Preprocessing](#preprocessing)
5. [Modelization](#modelization)
6. [Results](#results)

# Problem description

<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_desc.png" width="1200">

[Back to table of contents](#table-of-contents)

# Data analysis overview

## Number of images per classe

<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_class.png" width="600">

1. Minimum number of images: 148 (redbone)
2. Maximum number: 252 (Maltese_dog)
3. Average number: 171.5

## Distribution per image size

<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_size.png" width="600">

Most of the time, the images are of medium size

[Back to table of contents](#table-of-contents)

# Features engineering

<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_features.png" width="600">

The extraction of features is done automatically thanks to the use of a convolutional neural network.
It consists of a stack of convolution and pooling layers to end up with one or more completely connected layers.

[Back to table of contents](#table-of-contents)

# Preprocessing

## Equalization

### Before

<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_equalization_b.png" width="600">

### After

<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_equalization_a.png" width="600">

## Data augmentation

<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_augmentation.png" width="600">

Data augmentation is a technique used to increase the amount of data by adding slightly modified copies of existing data.
It helps reduce overfitting.

[Back to table of contents](#table-of-contents)

# Modelization

## My own CNN

Model composed of 20 layers and inspired by Inception V1

<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_own.png" width="1200">

## Results


<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_own_res.png" width="600">

## Pretrained models

1. Inception V3
2. ResNet50
3. VGG16

[Back to table of contents](#table-of-contents)

# Results

<img src="https://raw.githubusercontent.com/jamesbarthelemy/images/main/p6_result.png" width="600">

The selected model has an accuracy score of 0.8197.
The overfitting can be further reduced by using data augmentation during a second iteration.

[Back to table of contents](#table-of-contents)
