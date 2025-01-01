# 🚗 **Self-Driving Car Project** 🤖

## 📖 **Overview**

This project leverages deep learning techniques, particularly Convolutional Neural Networks (CNNs), to predict steering angles for a self-driving car based on input camera images. The model is inspired by NVIDIA's end-to-end self-driving architecture, enabling the car to make decisions in real-time using visual data. 

The project includes steps for data collection, model training, testing, and using TensorBoard for visualizing training progress. The entire codebase is tracked with **Git** and hosted on **GitHub**.

---

## 🛠️ **Technologies Used**

- **Python** 🐍: The programming language used for the implementation.
- **PyTorch** 🔥: Deep learning framework used for building and training the model.
- **TensorBoard** 📊: Visualization tool for tracking the training metrics.
- **Git/GitHub** 💻: Version control system for managing the project code.
- **OpenCV** 🎥: Used for image processing.
- **Pandas** 📊: For handling data files (CSV) and logs.
- **Matplotlib** 📉: For visualizing performance metrics and graphs.

---

## 📦 **Dataset**

The dataset consists of images captured by the car's center-mounted camera, paired with corresponding steering angle labels. This data is used for training the model to predict the car's steering commands based on the images.

### Key Files:
- **`steering_angles.csv`** 📑: Contains the steering angles for each image.
- **`driving_log.csv`** 📜: Log file containing additional car sensor data.
- **`Training_data/`** 📂: Folder containing all images collected from the center camera of the car.

---

## 🧠 **Model Architecture**

The model follows the **NVIDIA self-driving car architecture**, a CNN that processes the input images to predict a single steering angle for each image. Here's a brief breakdown:

### Layers:
- **Convolutional Layers** 🧩: Extract visual features from input images.
- **Fully Connected Layers** ⚡: Process the extracted features and predict the steering angle.
- **Output Layer** ⬅️: A single output neuron that provides the steering angle.

### Input and Output:

- **Input**: Image size `[batch_size, 3, 66, 200]` (RGB images)
- **Output**: Steering angle (continuous value)

---

## 🚀 **Training the Model**

Training involves several steps, including:

1. **Data Loading** 📥: Loading the images and labels.
2. **Preprocessing** 🔧: Normalizing and resizing images for the model.
3. **Training** 🎓: The model is trained using **Mean Absolute Error (MAE)** as the loss function, optimized with **Adam**.
4. **Visualization** 📈: Training metrics like loss and MAE are logged to **TensorBoard**.

---

## 📊 **Evaluation**

After training, the model is evaluated using the test dataset. The evaluation involves:

- Monitoring both **loss** and **accuracy** (MAE) during both training and testing.
- **TensorBoard** visualization of performance metrics.

To visualize the metrics using TensorBoard, run:

```bash
tensorboard --logdir=./logs
```

Then open `http://localhost:6006` in your browser.

---

## 🛠️ **Setup and Installation**

Follow these steps to set up the project on your local machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JaganFoundr/Self_Driving_Car.git
   cd Self_Driving_Car
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure that the dataset is available**:
   - Download and store the dataset images and CSV files in the `Training_data` folder.

---

## ▶️ **Running the Model**

To train and evaluate the model, simply run:

```bash
python train.py
```

This will start the training process, and you’ll see metrics printed in the terminal. 

For real-time visualization with **TensorBoard**, run:

```bash
tensorboard --logdir=./logs
```

Open the link `http://localhost:6006` in your browser to see the training curves!

---

## 🌱 **Future Work**

While this model is a great start, there are always opportunities for further enhancement:

- **Improve Model Architecture** 🚀: Implement more advanced models such as DAVE-2 or other newer architectures.
- **Data Augmentation** 🌍: Expand the dataset with augmented images (e.g., flipping, rotating) for better generalization.
- **Real-Time Testing** 🎮: Integrate the model into a real-world car simulator or physical vehicle equipped with a camera.
- **Sensor Fusion** 🛰️: Combine camera data with LiDAR or GPS data to enhance decision-making for the model.

---

## 📝 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for more details.

---

Feel free to reach out if you need help, and let’s continue working towards creating a safer and smarter self-driving car! 🚘🤖
