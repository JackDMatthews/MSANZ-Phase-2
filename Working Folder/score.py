import json, os, joblib
import numpy as np

#I could improve this to just give the results, but that means creating the endpoint again, and this works fine as is
classes = {'A': "A", 'B': "B", 'C': "C", 'D' : "D"}

def init():
    # Loads the model
    global model
    model_path = "MarketClassifier.pkl"
    full_model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_path)
    model = joblib.load(full_model_path)

def run(request):
    # Loads the input data, runs the model on it, and returns its predictions
    data = json.loads(request)
    data = np.array(data["data"])
    result = model.predict(data)
    return [classes.get(key) for key in result] 