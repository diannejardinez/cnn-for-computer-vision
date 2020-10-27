# Convolutional Neural Network (CNN) for computer vision
CNN is a type of deep learning neural network that is very effective at computer vision. This repository will be a group of small projects regarding CNN for Computer vision using Keras and Tensorflow




### Table of contents
* [Keras Applications - Static Images](#1-keras-applications---static-images)
* [MNIST classification digits](#2-keras-built-in-small-datasets-mnist-digits-classification)
* [MNIST classification Sign Language](#3-kaggle-dataset-mnist-classification-sign-language)


---

### 1. Keras Applications - Static Images
> Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning - [Source](https://keras.io/api/applications)

#### Project Summary
Printed out top 5 prediction labels from models for 10 images. The results below states when the model predicted correctly out of the top 5.


**Images used**: (1) Pembroke Welsh Corgi, (2) Cocker Spaniel, (3) Giant Panda, (4) Hamster, (5) Hedgehog, (6) Brittany Spaniel, (7) Macaw, (8) Tabby cat, (9) Red-eyed Tree Frog, (10) French Bulldog 



Model | Parameters | Known Top-5 Accuracy |Image Results
------|------------|----------------------|------------- 
VGG19 | 143,667,240 | 0.900 | 7: top 1<br>2: top 2<br>1: no top 5
ResNet152 | 60,419,944| 0.931 | 8: top 1<br>1: top 2<br>1: no top 5
Xception | 22,910,480 | 0.945 | 8: top 1 <br>1: top 4<br>1: not top 5
InceptionResNetV2 | 55,873,736 | 0.953 | 8: top 1<br>1: top 4<br>1: no top 5
NASNetLarge | 88,949,818 | 0.960 | 8: top 1<br>1: top 4<br>1: no top 5


#### Analysis: Ranking Models
1. ResNet152<br>
2. Xception, InceptionResNetV2, NASNetLarge<br>
3. VGG19

---

### 2. Keras Built-in small Datasets: MNIST digits classification 
> The tf.keras.datasets module provide a few toy datasets (already-vectorized, in Numpy format) that can be used for debugging a model or creating simple code examples - [Source](https://keras.io/api/datasets)

#### Project Summary
Creating a model based on the MNIST Dataset of grayscale image data with shapes and testing our model on test and real data.


#### Dependencies

Basic imports:
```
import numpy as np
import matplotlib.pyplot as plt
import os
```


ML imports:
```
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from IPython.display import Image, SVG
```



#### Steps
##### Load the Data
- Use `tf.keras.datasets` to load the 60,000 28x28 grayscale images of the 10 digits [MNIST digits classification dataset](https://keras.io/api/datasets/mnist/)
- This dataset has 2 splits: 'train' and 'test'

##### Explore datatset
```
There are 10,000 images in the test set
There are 60,000 images in the training set
```


##### Create Pipeline
- Normalize the training set and testing set
- Use `MinMaxScaler` to scale the numerical data
- Resize the training and testing from 28x28 to 784 pixels
- Shuffle and batch the data with `batch_size = 32`


##### Build and Train the Classifier
- Create a tf.keras Sequential model with the following layers:
	- Dense layer with 100 neurons and a `relu` activation function
	- Dropout layers with the dropout rate = 0.25 and 0.5 after each Dense layer. This prevents overfitting and produced the best results.
- Train the classifier
	```
	model.compile(loss = 'categorical_crossentropy',
	              optimizer = SGD(0.01),
	              metrics = ['accuracy'])
	```

- Fit the model
	- Use `10 Epochs`
	- Used `X_test, y_test` as the validation data
	
	```
	history = model.fit(X_train,
	                    y_train,
	                    batch_size = batch_size,
	                    epochs = epochs,
	                    verbose = 1,
		                validation_data = (X_test, y_test))
	```
- Test the model and print the loss and accuracy values
	```
	Loss on the TEST Set: 0.11469
	Accuracy on the TEST Set: 0.9645
	```

- Save the model
- Plot the loss and accuracy values achieved during training the the training and validation set
![](https://github.com/diannejardinez/cnn-for-computer-vision/blob/main/CNN-MNIST-digits/Images/loss_and_accuracy_charts.png)


#### Analysis - Model
- Output prediction example of model(left) from image data(right)
![](https://github.com/diannejardinez/cnn-for-computer-vision/blob/main/CNN-MNIST-digits/Images/output_1.png)

- Output prediction example of model(left) from image data(right)
![](https://github.com/diannejardinez/cnn-for-computer-vision/blob/main/CNN-MNIST-digits/Images/output_3.png)


---

### 3. Kaggle dataset: MNIST classification Sign Language

#### Project Summary
Creating a model based on the Kaggle Dataset of an American Sign Language letter database of 24 hand gestures representing letters (excluding J and Z which require motion) and testing our model on test and real data.

#### Analysis
- Accuracy and Loss Chart
	![](https://github.com/diannejardinez/cnn-for-computer-vision/blob/main/CNN-MNIST-Sign-Language/Images/test_accuracyandloss_chart.png)
- Test Accuracy score: **0.7726**

- Webcam Test
	- Top row (Sign language C, model correct)
	- Bottom row (Sign language A, model incorrect)
	![](https://github.com/diannejardinez/cnn-for-computer-vision/blob/main/CNN-MNIST-Sign-Language/Images/webcam_test.png)

#### Limitations
- Model trained on sign language letters and not words or phrases
- Camera may not be getting correct angle that mirrors Sign Language MNIST dataset 



