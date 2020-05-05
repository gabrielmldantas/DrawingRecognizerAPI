# DrawingRecognizerAPI

API and training program for the Drawing Recognizer app.

Install dependencies with pip install -r requirements.txt and run with flask run.
There's a trained CNN available for direct use. For new trainings, you need to download the required datasets
from Google Quick, Draw! project and place them on the resources/datasets directory.

cnn.py contains the CNN training program and generates the CSV and PNG files for analysis.
traditional_model.py contains the Decision Tree Classifier training program and generates the CSV and PNG files for analysis.
app.py contains the Flask API.
stats.py executes the Shapiro-Wilk and Wilcoxon tests.
