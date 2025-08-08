# Face Mask Detection Using CNN

This project implements a deep learning pipeline for detecting whether a person is wearing a face mask or not, using a custom Convolutional Neural Network (CNN) built and trained from scratch. The solution covers image preprocessing, model design, training, and result visualization.

---

## 1. Dataset Introduction

The dataset used in this project consists of two categories: "with_mask" and "without_mask". Each category contains images of people either wearing or not wearing a face mask. The images are processed and used to train a binary classifier.

- **Example dataset URL:** [https://www.kaggle.com/datasets/omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

---

## 2. Image Processing Code and Explanation

The following code loads and preprocesses the images, resizing them and ensuring consistent color channels:

```python
# Convert the with mask image to numpy array
with_mask_path = '/content/data/with_mask/'

data = []

for img_file in with_mask_files:
    image = Image.open(with_mask_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB') # Prevent the mix of grayscale and RGB
    image = np.array(image)
    data.append(image)

without_mask_path = '/content/data/without_mask/'

for img_file in without_mask_files:
    image = Image.open(without_mask_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB') # Prevent the mix of grayscale and RGB
    image = np.array(image)
    data.append(image)
```
**Explanation:**
- Images are loaded from their respective folders.
- All images are resized to 128x128 pixels to maintain uniform input shape for the CNN.
- Images are converted to RGB format to avoid inconsistencies caused by grayscale images.
- Images are converted to numpy arrays and appended to the dataset.

---

## 3. Model Architecture with Detailed Layer Comments

Below is the CNN model used for face mask detection, with layer-by-layer explanation:

```python
num_of_classes = 2
model = keras.Sequential()

# First convolutional layer: extracts 32 feature maps using 3x3 kernels, applies ReLU activation
model.add(keras.layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (128,128,3)))

# First max pooling layer: reduces spatial dimensions, helps with translation invariance
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

# Second convolutional layer: extracts 64 feature maps for deeper features
model.add(keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'))

# Second max pooling layer: further reduces spatial dimensions
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

# Flatten layer: flattens the 3D feature maps to 1D vector for dense layers
model.add(keras.layers.Flatten())

# First dense (fully connected) layer: 128 neurons, ReLU activation for non-linearity
model.add(keras.layers.Dense(128, activation = 'relu'))

# Dropout layer: randomly drops 50% of the neurons during training to reduce overfitting
model.add(keras.layers.Dropout(0.5))

# Second dense layer: 64 neurons, ReLU activation
model.add(keras.layers.Dense(64, activation = 'relu'))

# Second dropout layer: again, 50% dropout rate for regularization
model.add(keras.layers.Dropout(0.5))

# Output layer: two neurons (for binary classification), sigmoid activation
model.add(keras.layers.Dense(num_of_classes, activation = 'sigmoid'))

# Compile the model: using Adam optimizer and sparse categorical crossentropy loss
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['acc']
)
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=10)
```

---

## 4. Training Results

```
Epoch 1/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 14s 48ms/step - acc: 0.6600 - loss: 0.6825 - val_acc: 0.8496 - val_loss: 0.3160
Epoch 2/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 11s 16ms/step - acc: 0.8754 - loss: 0.3036 - val_acc: 0.8810 - val_loss: 0.2847
Epoch 3/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 3s 18ms/step - acc: 0.8900 - loss: 0.2659 - val_acc: 0.8909 - val_loss: 0.2521
Epoch 4/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - acc: 0.9267 - loss: 0.1903 - val_acc: 0.9041 - val_loss: 0.2742
Epoch 5/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - acc: 0.9269 - loss: 0.1738 - val_acc: 0.9025 - val_loss: 0.2526
Epoch 6/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - acc: 0.9421 - loss: 0.1481 - val_acc: 0.9091 - val_loss: 0.2621
Epoch 7/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 4s 20ms/step - acc: 0.9472 - loss: 0.1347 - val_acc: 0.9256 - val_loss: 0.2477
Epoch 8/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - acc: 0.9510 - loss: 0.1188 - val_acc: 0.9174 - val_loss: 0.2560
Epoch 9/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - acc: 0.9548 - loss: 0.1025 - val_acc: 0.9058 - val_loss: 0.2692
Epoch 10/10
170/170 ━━━━━━━━━━━━━━━━━━━━ 4s 23ms/step - acc: 0.9630 - loss: 0.0996 - val_acc: 0.9273 - val_loss: 0.3212

48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 24ms/step - acc: 0.9424 - loss: 0.1644
Test Accuracy = 0.9365
```

---

## 5. Training Result Visualization
- Below is the plot of training and validation accuracy over epochs:
<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/d63faeac-efd9-421b-af8d-47971783ff87" />

- Below is the plot of training and validation loss over epochs:
<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/a145e541-385e-4cca-9a55-b5fc15de412a" />

---

## 6. Conclusion and Analysis

- The CNN model demonstrates strong performance on face mask detection, with training accuracy climbing above 96% and test accuracy reaching approximately 94%.
- The loss curves show effective learning and convergence, with training loss decreasing steadily. The validation loss fluctuates slightly in later epochs, possibly due to minor overfitting, but overall remains low.
- The model generalizes well from training to validation and test sets, suggesting that the preprocessing and regularization strategies (dropout, pooling, color conversion) are effective.
- This pipeline can be directly adapted or further enhanced for other binary or multi-class image classification tasks in a similar domain.

---
> All model code, logs, and result plots are based on the original notebook and project files.  
> For any questions or suggestions, please open an issue.
