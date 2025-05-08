# Tomato-leaves-Disease-Detection
Developed a model to differentiate Healthy and Unhealthy Tomato Leaves
🍅 Tomato Leaf Disease Classification (Deep Learning with TensorFlow)

This project uses a Convolutional Neural Network (CNN) to classify images of tomato leaves into 10 disease categories using TensorFlow and Keras. The model is trained on a dataset of over 16,000 labeled images, with support for data augmentation, GPU acceleration, checkpointing, and automatic resume training.

=>Dataset
	•	The dataset contains 16,011 tomato leaf images across 10 classes, including diseases like:
	•	Tomato Mosaic Virus
	•	Late Blight
	•	Early Blight
	•	Bacterial Spot
	•	Healthy leaves
	•	Images are organized into folders (one per class) and loaded using image_dataset_from_directory.


=>Features
	•	CNN model built with Keras Sequential API
	•	Rescaling and augmentation with Resizing, RandomFlip, RandomRotation, etc.
	•	Regularization using:
	•	Dropout
	•	BatchNormalization
	•	L2 weight regularization
	• Model Checkpointing: saves the best model using val_loss
	• Resume training automatically from last checkpoint
	• Early stopping to prevent overfitting
	•	GPU support for Apple M1 Metal or other platforms


=> Model Architecture
	•	3 Convolutional layers (Conv2D) with ReLU and MaxPooling
	•	Batch normalization
	•	GlobalAveragePooling
	•	Dense layers with softmax output
	•	Dropout layers to reduce overfitting

 =>Training Configuration
	•	Optimizer: Adam (learning_rate=0.0001)
	•	Loss: Sparse Categorical Crossentropy
	•	Epochs: 25
	•	Batch size: 32
	•	Callbacks:
	•	ModelCheckpoint to save the best model
	•	EarlyStopping to stop when no improvement
