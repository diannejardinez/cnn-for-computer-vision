# Convolutional Neural Network (CNN) for computer vision
CNN is a type of deep learning neural network that is very effective at computer vision. This repository will be a group of small projects regarding CNN for Computer vision using Keras and Tensorflow


### Keras Applications - Static Images
> Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning - [Source](https://keras.io/api/applications/)

**Available Models used**

Printed out top 5 prediction labels from models for 10 images. The results below states when the model predicted correctly out of the top 5.
<br>Images: (1) Pembroke Welsh Corgi, (2) Cocker Spaniel, (3) Giant Panda, (4) Hamster, (5) Hedgehog, (6) Brittany Spaniel, (7) Macaw, (8) Tabby cat, (9) Red-eyed Tree Frog, (10) French Bulldog 



Model | Parameters | Known Top-5 Accuracy |Image Results
------|------------|--------
Xception | 22,910,480 | 0.945 | 
VGG19 | 143,667,240 | 0.900 |
ResNet152 | 60,419,944| 0.931 |
InceptionResNetV2 | 55,873,736 | 0.953 |
NASNetLarge | 88,949,818 | 0.960 |
