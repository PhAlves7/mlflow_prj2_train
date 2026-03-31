import mlflow 
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile, Query
import joblib
import boto3
from dotenv import load_dotenv
import os
from datetime import date, timedelta, datetime
from io import StringIO
from enum import Enum


load_dotenv()

MLFLOW_TRACKING_URI = r"https://phahugg-demo-mlflow.hf.space"

"""
Prediction du doctolib 
"""
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set your variables for your environment
EXPERIMENT_NAME="doctolib_simplified"
# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

#print(type(predictionFeatures), predictionFeatures)
# Read data 
data = pd.read_csv("doctolib_simplified_dataset_01.csv")

logged_model = "models:/Doctolib_simplified_model@challenger"


# # Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
print('loaded model')
prediction = loaded_model.predict(data)
print(prediction)


# Format response
response = {"doctolib": prediction.tolist()}
resp_df = pd.DataFrame(response)

# resp_df.to_csv("predi.csv")
resp_df.to_csv('doctolib_simplified_dataset_01_predictions.csv', index=None)
print(response)