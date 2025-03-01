# Sign Language Digit Recognition

This project demonstrates how to recognize sign language gestures for digits 1-10 using both Convolutional Neural Networks (CNNs) and Fully Connected Neural Networks (FCNNs). It also provides a visualization tool to understand the inner workings of a trained CNN.

## Dataset

The dataset used is the "Sign Language Digits Dataset" by [Arda Mavi](https://github.com/ardamavi/Sign-Language-Digits-Dataset). It consists of 2062 images of hand gestures representing digits 1-10.

* **Dataset files:**
    * `dataset/X.npy`: Contains the images.
    * `dataset/Y.npy`: Contains the labels.

**Important Note:** The original dataset had label inconsistencies that have been addressed in this project. For more information, refer to the [dataset discussion](https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset/discussion/57074).

## Project Structure

* `cnn_sign_language.ipynb`: Contains the code for training and evaluating a Convolutional Neural Network (CNN).
* `fcnn_sign_language.ipynb`: Contains the code for training and evaluating a Fully Connected Neural Network (FCNN).
* `visualization_cnn.ipynb`: Contains the code for visualizing the feature maps and activations of a trained CNN.
* `dataset/`: Contains the dataset files (`X.npy` and `Y.npy`).
* `images/`: Contains images as a GIF.
* `model.keras`: The saved trained CNN model.

## CNN Model Details

* **Architecture:**
    * Input: 64x64 grayscale images.
    * Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation.
    * MaxPooling Layer 1: 2x2 pooling.
    * Convolutional Layer 2: 64 filters, 3x3 kernel, ReLU activation.
    * MaxPooling Layer 2: 2x2 pooling.
    * Convolutional Layer 3: 128 filters, 3x3 kernel, ReLU activation.
    * MaxPooling Layer 3: 2x2 pooling.
    * Flatten Layer: Converts 2D feature maps to a 1D vector.
    * Dense Layer 1: 128 neurons, ReLU activation.
    * Dropout Layer: 0.5 dropout rate.
    * Output Layer: 10 neurons, Softmax activation.
* **Optimizer:** rmsprop
* **Epochs:** 20

## FCNN Model Details

* **Architecture:**
    * Input: Flattened 64x64 grayscale images (4096 neurons).
    * Dense Layer 1: 128 neurons, ReLU activation.
    * Dense Layer 2: 128 neurons, ReLU activation.
    * Output Layer: 10 neurons, Softmax activation.
* **Optimizer:** adam
* **Epochs:** 50

## Results

* **CNN Model:**
    * Loss: Approximately 0.22
    * Accuracy: Approximately 92.5%
* **FCNN Model:**
    * Loss: Approximately 0.45
    * Accuracy: Approximately 87.4%

## Usage

1.  **Dataset:** Ensure that the `dataset/X.npy` and `dataset/Y.npy` files are in the `dataset` folder.
2.  **CNN Model:**
    * Open and run `cnn_sign_language.ipynb` in Jupyter Notebook to train and evaluate the CNN model.
    * The trained model will be saved as `model.keras`.
3.  **FCNN Model:**
    * Open and run `fcnn_sign_language.ipynb` in Jupyter Notebook to train and evaluate the FCNN model.
4.  **CNN Visualization:**
    * Ensure that `model.keras` is in the same directory.
    * Open and run `visualization_cnn.ipynb` in Jupyter Notebook to visualize the feature maps and activations of the trained CNN model.

## Visualizations

The `visualization_cnn.ipynb` script provides visualizations of:

* Feature maps from convolutional and pooling layers.
* Activations from the dense layer.
* Output probabilities for each class.

**CNN Visualization:**

The following animation shows how an input image is transformed as it passes through the layers of the CNN.

![CNN Visualization](images/images.gif)

## Dependencies

* Python 3.x
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow/Keras

To install the required packages, you can use pip:

```bash
pip install numpy matplotlib scikit-learn tensorflow