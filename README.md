Rock vs Mine Prediction


Predicting whether a sonar signal represents a rock or a mine using machine learning models.
📋 Table of Contents

    Project Overview
    Technologies Used
    Dataset
    Installation
    Usage
    Model Evaluation
    Results
    Contributing
    License

📖 Project Overview

This project involves classifying sonar signals to determine whether they originate from rocks or metal mines. By analyzing frequency and amplitude data, we aim to develop a robust predictive model using supervised machine learning.
💻 Technologies Used

    Python 3.12
    Libraries:
        Pandas
        NumPy
        Matplotlib
        Seaborn
        Scikit-Learn

📂 Dataset

The dataset is sourced from the UCI Machine Learning Repository.
Dataset Features:

    Input Variables: 60 sonar signal frequency readings.
    Target Variable:
        R: Represents rocks.
        M: Represents mines.

🛠 Installation

    Clone this repository:

git clone https://github.com/yourusername/RockVsMinePrediction.git  
cd RockVsMinePrediction  

Install required dependencies:

    pip install -r requirements.txt  

    Ensure the dataset is placed in the data/ folder.

🚀 Usage

    Preprocess Data: Run the script to clean and preprocess the dataset:

python preprocess.py  

Train the Model: Use the training script:

python train_model.py  

Make Predictions: Predict new signals:

    python predict.py --input new_data.csv --output predictions.csv  

    Notebook Option: Open RockVsMinePrediction.ipynb to explore the data analysis and model training steps interactively.

📈 Model Evaluation

We performed cross-validation on various machine learning models and evaluated their performance using the F1-score metric. The models tested include:

    Logistic Regression
    Decision Tree
    Random Forest
    Support Vector Machine (SVM)
    k-Nearest Neighbors (kNN)
    Gradient Boosting

F1-Score Comparison

A comparison graph showcasing the F1-scores of different models is included in the results.
📊 Results


The F1-score analysis revealed that K Nearest Neighbours(KNN-3) provided the best balance of precision and recall for this dataset.
🤝 Contributing

Contributions are welcome!

    Fork the repository.
    Create a new branch: git checkout -b feature-name.
    Commit your changes: git commit -m 'Add feature'.
    Push to the branch: git push origin feature-name.
    Submit a pull request.

📝 License

This project is licensed under the MIT License.
