# Anomaly Detection in IoT Sensor Machine Learning Model GUI

This GUI application facilitates the anomaly detection process in IoT sensor data using machine learning models. It allows users to load their CSV data, preprocess it, train machine learning models (Logistic Regression and Isolation Forest), and visualize the results.

## Features

- Load CSV data file containing IoT sensor data.
- Preprocess the data by removing unnecessary columns, encoding categorical variables, and splitting into training and testing sets.
- Train machine learning models (Logistic Regression and Isolation Forest) on the preprocessed data.
- Evaluate model performance using accuracy scores and classification reports.
- Visualize the distribution of machine statuses using a countplot.
- Display anomaly detection results and classification reports in a separate window.
- Quit the application.

## Prerequisites

- Python 3.x
- Required Python libraries: `csv`, `pandas`, `seaborn`, `numpy`, `matplotlib`, `tkinter`, `scikit-learn`

## Installation

1. Clone or download this repository.
2. Install the required Python libraries using pip:



3. Run the Python script `anomaly_detection_gui.py`.

## Usage

1. Launch the GUI application.
2. Click on "Load Data and Run Model" to load your CSV data and start the anomaly detection process.
3. After processing, the application will display the results in a separate window, including accuracy scores, classification reports, and a countplot of machine statuses.
4. Close the separate window to return to the main GUI.
5. Click on "Quit" to exit the application.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


