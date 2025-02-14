# Spam Detection App

A Streamlit-based web application that classifies text messages as spam or ham using a machine learning model trained on the SMS Spam Collection dataset.

## Features
	•	Machine Learning Pipeline: Uses TF-IDF vectorization and a Multinomial Naive Bayes classifier.
	•	Hyperparameter Tuning: Implements GridSearchCV for optimal performance.
	•	Interactive UI: Built with Streamlit for real-time predictions.
	•	Batch Prediction: Allows CSV uploads for multiple message classifications.
	•	Dynamic UI: Includes a reset button to clear input fields seamlessly.

## Technologies Used
	•	Python
	•	Scikit-learn
	•	Streamlit
	•	Pandas
	•	NumPy

## How to Run Locally
	1.	Clone the Repository:

git clone https://github.com/carasolomon/spam_detection.git

cd spam_detection


	2.	Install Dependencies:

pip install -r requirements.txt


	3.	Run the App:

streamlit run app.py

## Deployment

This app is ready for deployment on:
	•	Streamlit Cloud
	•	Heroku
	•	Render


