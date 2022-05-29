# Overview
The model combines the characteristics of URLs in the field of web attacks at the character level. It treats URLs, file paths, and registries as a short string. The model contains character-level embedding and a convolutional neural network to extract features, and the extracted features is passed onto the hidden layers for classification.

# Requirements
- Python 3.7 (tested under Python 3.7.0)

# Performance
![Accuracy and Loss Graph](https://github.com/SK9712/Detecting-Malicious-Url-Using-Character-level-CNN/blob/master/Graph/training_testing_accuracy_loss.png?raw=true)
