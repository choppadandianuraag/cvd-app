from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from fastapi.responses import JSONResponse
from typing import Literal, Annotated
import joblib
import pandas as pd
import os

app= FastAPI()

# Use relative path that works both locally and in Docker
model_path = os.path.join(os.path.dirname(__file__), 'xgb_grid.pkl')
model = joblib.load(model_path)

class Patient_input(BaseModel):
    age: Annotated[int, Field(...,gt=0, lt=100,description="Age of the patient in years")]
    gender: Annotated[Literal['male','female'],Field(...,description="Gender of the patient")]
    height: Annotated[float, Field(...,gt=0,lt=250,description='Height of the patient in cm')]
    weight: Annotated[float, Field(...,gt=0,lt=150,description='Weight of the patient in kg')]
    ap_hi: Annotated[int, Field(...,gt=0,lt=250,description='Systolic blood pressure')]
    ap_lo: Annotated[int, Field(...,gt=0,lt=150,description='Diastolic blood pressure')]
    cholesterol: Annotated[int, Field(...,ge=100,lt=260,description='Cholesterol level')]
    gluc: Annotated[int, Field(...,ge=50,lt=150,description='Glucose level')]
    smoke: Annotated[Literal['yes','no'],Field(...,description='Whether the patient smokes')]
    alco: Annotated[Literal['yes','no'],Field(...,description='Whether the patient consumes alcohol')]
    active: Annotated[Literal['yes','no'],Field(...,description ='Whether the patient is physically active')]

    @computed_field
    @property
    def bmi(self)-> float:
        height=self.height/100
        return (self.weight/(height*height))
    
    @computed_field
    @property
    def gender_en(self)-> int:
        return 1 if self.gender == 'male' else 0
    
    @computed_field
    @property
    def cholesterol_level(self)-> int:
        if self.cholesterol < 200:
            return 1
        elif 200 <= self.cholesterol <240:
            return 2
        else:
            return 3
    
    @computed_field
    @property
    def glucose_level(self)-> int:
        if self.gluc < 100:
            return 1
        elif 100 <= self.gluc <=125:
            return 2
        else:
            return 3
    
    @computed_field
    @property
    def smoking(self)-> int:
        return 1 if self.smoke == 'yes' else 0
    
    @computed_field
    @property
    def alcohol(self)-> int:
        return 1 if self.alco == 'yes' else 0
    
    @computed_field
    @property
    def activity(self)-> int:
        return 1 if self.active == 'yes' else 0
    
@app.get('/')
def intro():
    return JSONResponse(content={"message":"Welcome to Cardio Disease Prediction API"})

@app.post('/predict')
def predict_cardio(data: Patient_input):
    # Map cholesterol levels to categorical labels
    cholesterol_map = {1: 'normal', 2: 'high', 3: 'very high'}
    cholesterol_category = cholesterol_map[data.cholesterol_level]
    
    # Glucose levels remain as numbers but need to be strings for get_dummies
    gluc_category = str(data.glucose_level)

    input_data=pd.DataFrame([{
        'age':data.age,
        'gender':data.gender_en,
        'ap_hi': data.ap_hi,
        'ap_lo': data.ap_lo,
        'smoke': data.smoking,
        'alco': data.alcohol,
        'active': data.activity,
        'bmi':data.bmi,
        'cholesterol':cholesterol_category,
        'gluc':gluc_category
    }])
    input_data=pd.get_dummies(input_data,columns=['cholesterol','gluc'])

    # Define the expected feature columns from training
    trained_feature_columns = ['age', 'gender', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 
                               'bmi', 'cholesterol_high', 'cholesterol_normal', 
                               'cholesterol_very high', 'gluc_1', 'gluc_2', 'gluc_3']

    input_data = input_data.reindex(
    columns=trained_feature_columns,
    fill_value=0
    )

    # Get probability of positive class (class 1)
    pred_proba = model.predict_proba(input_data)[0]
    # pred_proba is an array like [prob_class_0, prob_class_1]
    # Get the probability of the positive class (cardiovascular disease)
    prob_positive = pred_proba[1]
    
    # Make prediction based on threshold
    prediction = 1 if prob_positive > 0.45 else 0

    return JSONResponse(status_code=200, content={'predicted_category': int(prediction)})
