# Camera Detection module for number Plate Recognition

## Package Installation Guide

Run the Following command in your terminal to install every required package and make sure you have pip installed in your system

```bash
pip install opencv-python numpy scikit-image tensorflow Pillow matplotlib pymongo requests
```

or you can also run the command below after cloning this repository

```bash
pip install -r requirements.txt
```

## Brief Description of the Detection module

The `most_similar_plate` function is designed to find the most similar license plate to a given target plate within a MongoDB collection. It connects to MongoDB using a provided URL and accesses the "fmg_personal" database and the "car_data" collection. The function retrieves all license plate data from the collection, excluding the MongoDB-generated "_id" field. It then employs the `difflib` library to find the closest match to the target plate, based on string similarity. The resulting most similar license plate is returned by the function. This mechanism is useful for identifying and associating similar license plates within a database, facilitating tasks like pattern recognition or vehicle tracking.

The `get_plate` function is designed to extract alphanumeric characters from an image of a car, specifically focusing on recognizing and interpreting number plates. The process involves several steps:

1. **Preprocessing:**
   - Read the input image and convert it to grayscale.
   - If the image has three channels (indicating it is in color), convert it to grayscale.
   - Apply Gaussian blur to the grayscale image to remove noise.
   - Use [Otsu's](https://en.wikipedia.org/wiki/Otsu%27s_method) thresholding to create a binary image.

2. **Region Detection:**
   - Label connected regions in the binary image to identify potential text-like regions.
   - Filter and select regions based on certain criteria such as aspect ratio, area, and position within the image.

3. **Grouping and Sorting:**
   - Sort the selected bounding boxes based on the y-coordinate of the top-left corner.
   - Group adjacent boxes that have a small vertical difference (within a specified threshold) into clusters.

4. **OCR (Optical Character Recognition):**
   - Load pre-trained models (`model_alpha_path` and `model_num_path`) for recognizing alphanumeric characters.
   - Iterate over the grouped bounding boxes, and for each box:
     - Crop the corresponding region from the image.
     - Resize the cropped region to a consistent size.
     - Use the OCR models to predict the character in the cropped region.

5. **Post-processing:**
   - Assemble the predicted characters into a string.
   - Call the `most_similar_plate` function to find the most similar plate in the MongoDB collection, based on the predicted characters.

6. **Exception Handling:**
   - The function is wrapped in a try-except block to handle any potential errors during the process.

The `get_plate` function serves as a comprehensive tool for extracting and interpreting license plate information from car images, making use of image processing techniques and machine learning models for character recognition.

The `check_consistency` function verifies whether the length of the provided text, presumably representing a detected number plate, matches a specified length.

The `capture_predict_and_call_api` function serves as the main loop of the application. It captures video frames from a source, potentially a webcam, utilizing the OpenCV library. Each frame is saved as an image file named "image.jpeg". The code attempts to extract the number plate from the captured image using the `get_plate()` function, which employs image processing techniques like Otsu thresholding and region labeling. The consistency of the detected number plate is then checked using the `check_consistency()` function.

If the detected number plate is consistent or a failsafe condition (failsafe equals 3) is met, the code constructs a payload and sends a POST request to the (fmg.backend)[https://github.com/FindMyGarage/fmg.backend] API endpoint using the `requests` library. Based on the API response, the code prints messages indicating whether the car has successfully entered, exited, is already present, or if there is no corresponding booking found. The process runs in a continuous loop with a delay of 5 seconds between iterations.

## Models(tensorflow)

The `model_alpha` code defines a Convolutional Neural Network (CNN) using TensorFlow's Keras API. This model is designed for recognizing alphanumeric characters from images, such as license plates. The architecture includes convolutional layers for feature extraction, max-pooling layers for spatial reduction, and fully connected layers for classification. The model is trained to output probabilities for 26 & 10 classes (assumed to be alphabets and digits respectively) based on input images with dimensions of 128x128 pixels and three color channels (RGB). This CNN architecture is suitable for tasks like license plate character recognition.

```bash
model_alpha = tf.keras.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Variable input shape
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(26, activation='softmax')  # Assuming 26 alphabets
])

model_num = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Assuming 10 digits
])
```

## Optimization

In order to enhance efficiency, the model underwent quantization using TensorFlow Model Optimization (TF-MOT). Quantization is a process that reduces the precision of the model's weights, leading to a smaller memory footprint and faster inference times. Specifically, we utilized the `tfmot.quantization.keras.quantize_model` function from the TensorFlow Model Optimization library. This allows for the creation of a quantized version of the original model, optimizing its performance while maintaining reasonable accuracy for tasks like license plate character recognition.
