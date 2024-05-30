# AIRDRAW

This project showcases the creation of a Convolutional Neural Network (CNN) model in PyTorch for detecting and classifying human hand positions and poses from a webcam feed. These detected positions and poses are then utilized to draw on a virtual board created with OpenCV.

To achieve real-time performance, we explored various approaches and opted for a hands-on methodology to deepen our understanding of deep learning models. Instead of relying on pre-built training pipelines, we constructed the models from scratch, including custom dataloaders, loss functions, and model architectures. While ultralytics' YOLOv8 pipeline provided exceptional results with minimal code, our goal was to gain practical experience through manual implementation.

Our best results were obtained using a model with ResNet50 as the backbone, complemented by custom final layers. This model achieved an mAP@0.5:0.05:0.95 score of 0.38 and an mAP@0.5 score of 0.68 on the HaGrid dataset (original repository: https://github.com/hukenovs/hagrid) using a YOLOv1-like loss function.

We also experimented with creating our own model inspired by YOLOv1, training it from scratch. However, due to suboptimal results, we reverted to using a pretrained backbone.

The notebooks used to build and train the model are included in the repository.

Despite limited resources for training such a large model, we successfully developed a robust and effective solution. Below is a snippet showcasing the working model in action.

![image](https://github.com/mohitydv09/airdraw/assets/101336175/f29c7d84-2e3b-4c0d-a5b0-eadfb9367a92)
