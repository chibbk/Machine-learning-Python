# Machine-learning-Python
Machine Learning models applied on an Excel dataset using Python on Google Colab.

The Breast Cancer Wisconsin Dataset consists of information about breast cancer tumours. It was created by Dr. William H. Wolberg. This dataset was created to assist researchers with classifying tumours as either M (Malignant) or B (Benign).
The dataset contains 31 predictor features and 1 target feature (diagnosis). The predictor features include:
- Radius (the mean of distances from the centre to points on the perimeter).
- Texture (the standard deviation of gray-scale values).
- Perimeter
- Area
- Smoothness (the local variation in radius lengths).
- Compactness (the perimeter^2 / area - 1.0).
- Concavity (the severity of concave portions of the contour).
- Concave points (the number of concave portions of the contour).
- Symmetry
- Fractal dimension ("coastline approximation" - 1).

The ID number feature has been removed as each ID is unique and therefore there cannot be a pattern between ID number and tumor diagnosis.

The data is normalized using a Z-score scaler.

Classification models are used to predict whether a given input would render B or M. Using the Numpy, Pandas, Matplotlib, and Sci-kit learn libraries, 4 different classifiers were employed:

- Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel.
- Support Vector Machine (SVM) with Polynomial kernel.
- Neural Network Multi-Layer Perceptron (MLP).
- Gaussian Naive Bayes.

The predicted values are then compared with the test values (samples of 30% of the data) to determine whether each classifier is accurate.

Each classifier is called 3 times with different random seeds(random_state=40, random_state=20, random_state=1) and the average of these calls is taken.

Lastly, Histograms are displayed comparing predicted values with test values. The most accurate classification model is identified.

Below are samples of the histogram plots:

![first](https://github.com/chibbk/Machine-learning-Python/assets/158145884/5275651f-633a-4797-87e4-682220f27d2b)
![second](https://github.com/chibbk/Machine-learning-Python/assets/158145884/edf32eac-91a1-4453-9bff-97108dddab27)
