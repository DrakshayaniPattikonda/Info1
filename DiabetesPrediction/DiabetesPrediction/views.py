from  django.shortcuts import render
from django.shortcuts import render
import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the dataset and train the model once at startup
data = pd.read_csv("C:/Users/91970/Desktop/diabeties/Diabetes-Prediction-using-Machine-Learning-main/Diabetes-Prediction-using-Machine-Learning-main/Data/diabetes.csv")

X = data.drop('Outcome', axis=1)  # Ensure correct column name
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)  # Increased iterations to ensure convergence
model.fit(x_train, y_train)

def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    try:
        # Extract input values safely
        val1 = float(request.GET.get('n1', 0))
        val2 = float(request.GET.get('n2', 0))
        val3 = float(request.GET.get('n3', 0))
        val4 = float(request.GET.get('n4', 0))
        val5 = float(request.GET.get('n5', 0))
        val6 = float(request.GET.get('n6', 0))
        val7 = float(request.GET.get('n7', 0))
        val8 = float(request.GET.get('n8', 0))

        # Make prediction
        pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

        # Determine result
        result_text = "Positive" if pred[0] == 1 else "Negative"

    except Exception as e:
        result_text = f"Error: {str(e)}"

    return render(request, "predict.html", {"result": result_text})
