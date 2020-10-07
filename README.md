# Convolutional Neural Network (CNN) for computer vision
CNN is a type of deep learning neural network that is very effective at computer vision. This repository will be a group of small projects regarding CNN for Computer vision using Keras and Tensorflow

---

### Keras Applications - Static Images
> Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning - [Source](https://keras.io/api/applications)

**Project Summary**
<br>Printed out top 5 prediction labels from models for 10 images. The results below states when the model predicted correctly out of the top 5.


**Images used**: (1) Pembroke Welsh Corgi, (2) Cocker Spaniel, (3) Giant Panda, (4) Hamster, (5) Hedgehog, (6) Brittany Spaniel, (7) Macaw, (8) Tabby cat, (9) Red-eyed Tree Frog, (10) French Bulldog 



Model | Parameters | Known Top-5 Accuracy |Image Results
------|------------|----------------------|------------- 
VGG19 | 143,667,240 | 0.900 | 7: top 1<br>2: top 2<br>1: no top 5
ResNet152 | 60,419,944| 0.931 | 8: top 1<br>1: top 2<br>1: no top 5
Xception | 22,910,480 | 0.945 | 8: top 1 <br>1: top 4<br>1: not top 5
InceptionResNetV2 | 55,873,736 | 0.953 | 8: top 1<br>1: top 4<br>1: no top 5
NASNetLarge | 88,949,818 | 0.960 | 8: top 1<br>1: top 4<br>1: no top 5


**Ranking Models**
<br>1. ResNet152<br>
2. Xception, InceptionResNetV2, NASNetLarge<br>
3. VGG19

---

### Keras Built-in small Datasets: MNIST digits classification 
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



