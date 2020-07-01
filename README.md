# Siamese Neural Network for One-Shot Learning of Hand Gestures using Minimal Datasets

This repository hosts the network training codes for the Siamese Neural Net using the ASL Dataset. The aim of this project is to check the affect of **dataset volumes** and **network depth** on Siamese Neural Net's performance.

### [Check out the Software-Requirements-Specification Document](https://github.com/sudoRicheek/Siamese-Net-One-Shot-Learning/blob/master/SRS%20One%20Shot%20Learning.pdf).

### [Check out the network training codes](https://github.com/sudoRicheek/Siamese-Net-One-Shot-Learning/blob/master/Siamese%20Neural%20Network%20for%20Gesture%20Classification.ipynb).


## Model Description

The process of learning good features for machine learning applications can be very computationally expensive and may prove difficult in cases where little data is available. So there comes the Siamese Neural Network to address some of these issues.  We build upon the deep learning framework, which uses many layers of non-linearities to capture invariances to transformation in the input space.

Here our standard model is a siamese convolutional neural network with L layers. We use exclusively ReLU units in the first L - 2 layers and sigmoidal units in the remaining layers. The network applies a ReLU activation function to the output feature maps, optionally followed by maxpooling with a filter size and stride of 2. Finally we use the weighted L_1 distance between the twin vectors of siamese legs and combine them with a sigmoid avtivation, to get the input similarities.

Here I have used **Adam optimizer** with (lr = 1e-3, beta_1 = 0.9, beta_2=0.999), and received healthy results, but surely this could use some more finetuning and introducing a learning schedule seems a fair schoice. And the loss has been taken to be "binary_crossentropy".

**Gist of a Siamese Architecture :**

<p align = "center">
  <img src = "https://github.com/sudoRicheek/Siamese-Net-One-Shot-Learning/blob/master/ReadmeImages/siamese_nn.png" title = "siamese" width = "512" />
</p>

## Results

### Train Bench :
* **OS : Windows**
* **GPU : GTX 1060 6GB** 
* **CPU : i7 6700HQ ~2.60GHz**
* **RAM : 16GB**

### What the dataset images/gestures look like :

These are single channeled 28 x 28 images and here's what they look like :

<p align="center">
  <img src="/ReadmeImages/rand1.png" width="150" />
  <img src="/ReadmeImages/rand2.png" width="150" /> 
  <img src="/ReadmeImages/rand3.png" width="150" /> 
</p>

**Let's see the results of training the models over different dataset volumes :**

* With half the dataset : ```TRAIN_SET_PER_CLASS : 500 | VALIDATION_SET_PER_CLASS : 100 | TEST_SET_PER_CLASS : 200```
  Convnet Architecture : 
  ```
  (Conv2D(64,(7,7),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
  (MaxPooling2D())
  (Conv2D(128,(4,4),activation='relu',
                     kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
  (MaxPooling2D())
  (Conv2D(256,(3,3),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init,padding='same'))
  (Flatten())
  (Dense(2048,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
  ```
  <p align="center">
    <img src="/AccuracyCurves/half_val.png" width="386" />
    <img src="/AccuracyCurves/half_test.png" width="386" /> 
  </p>

* With one-fourth the dataset : ```TRAIN_SET_PER_CLASS : 250 | VALIDATION_SET_PER_CLASS : 75 | TEST_SET_PER_CLASS : 200```
  Convnet Architecture :
  ```
  (Conv2D(64,(7,7),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
  (MaxPooling2D())
  (Conv2D(128,(4,4),activation='relu',
                      kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
  (MaxPooling2D())
  (Conv2D(256,(3,3),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init,padding='same'))
  (Flatten())
  (Dense(2048,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
  ```
  <p align="center">
    <img src="/AccuracyCurves/one_fourth_val.png" width="386" />
    <img src="/AccuracyCurves/one_fourth_on_test.png" width="386" /> 
  </p>

* With one-eighth the dataset : ```TRAIN_SET_PER_CLASS : 125 | VALIDATION_SET_PER_CLASS : 50 | TEST_SET_PER_CLASS : 200```
  Convnet Architecture :
  ```
  (Conv2D(64,(7,7),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
  (MaxPooling2D())
  (Conv2D(128,(4,4),activation='relu',
                     kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
  (MaxPooling2D())
  (Conv2D(256,(3,3),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init,padding='same'))
  (Flatten())
  (Dense(1024,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
  ```
  <p align="center">
    <img src="/AccuracyCurves/one_eight_val.png" width="386" />
    <img src="/AccuracyCurves/one_eight_test.png" width="386" /> 
  </p>

* With 30 images per class the dataset : ```TRAIN_SET_PER_CLASS : 30 | VALIDATION_SET_PER_CLASS : 50 | TEST_SET_PER_CLASS : 200```
  Convnet Architecture :
  ```
  (Conv2D(64,(7,7),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
  (MaxPooling2D())
  (Conv2D(128,(4,4),activation='relu',
                     kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
  (MaxPooling2D())
  (Conv2D(128,(3,3),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init,padding='same'))
  (Flatten())
  (Dense(1024,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
  ```
  <p align="center">
    <img src="/AccuracyCurves/30_val.png" width="386" />
    <img src="/AccuracyCurves/30_test.png" width="386" /> 
  </p>
  
We see that the **affect** of data is not very pronounced between the ```one-half, one-quarter and one-eighth dataset sizes.```. But as soon as we drop the dataset size to ```30 IMAGES PER_CLASS``` it begins to affect the model adversely and produces unreliable results. So there must be a certain **dataset volume threshold** after which the model performance begins to get affected in a more noticeable way. It needs more exploration and more results to check if we can actually predict the **threshold** and improve our understanding of **Deep Siamese Networks**. :nerd_face: 
  
## References

* **[SOTA Siamese Neural Network on Ominglot Dataset.](https://paperswithcode.com/paper/siamese-neural-networks-for-one-shot-image)**

* **Deep Learning with Python by Francois Chollet**

* **[This GitHub Repository helped a lot.](https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning)**

## Author

Created with :heart: by Richeek Das **a.k.a.** **sudoRicheek**. **MIT Open-Source. June 2020.**
