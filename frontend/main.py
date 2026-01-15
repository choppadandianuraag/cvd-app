import streamlit as st
import requests
import os

API_URL = os.getenv("FASTAPI_URL")
st.title('Cardio Vascular Disease Prediction')

age= st.number_input('Enter Age',min_value=1,max_value=100)
gender=st.selectbox('Select your gender',options=['male','female'])
height=st.number_input('Enter height in cm',min_value=50,max_value=200)
weight=st.number_input('Enter weight in kg',min_value=10,max_value=200)
ap_hi=st.number_input('Enter Systolic blood pressure',min_value=50,max_value=250)
ap_lo=st.number_input('Enter Diastolic blood pressure',min_value=0,max_value=150)
cholesterol=st.number_input('Enter Cholesterol level',min_value=100,max_value=260)
gluc=st.number_input('Enter Glucose level',min_value=50,max_value=150)
smoke=st.selectbox('Do you smoke?',options=['yes','no'])
alco=st.selectbox('Do you consume alcohol?',options=['yes','no'])
active=st.selectbox('Are you physically active?',options=['yes','no']) 


if st.button('Predict'):
    input_data={
        'age':age,
        'gender':gender,
        'height':height,
        'weight':weight,
        'ap_hi':ap_hi,
        'ap_lo':ap_lo,
        'cholesterol':cholesterol,
        'gluc':gluc,
        'smoke':smoke,
        'alco':alco,
        'active':active
    }

    try:
        response=requests.post(f"{API_URL}/predict",json=input_data)
        
        # Check if response is successful first
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('predicted_category')
            
            if prediction is not None:
                if prediction == 1:
                    st.error(f"⚠️ High Risk: Cardiovascular disease detected")
                else:
                    st.success(f"✅ Low Risk: No cardiovascular disease detected")
                # st.write(f"Prediction: **{prediction}**")
            else:
                st.error("Unexpected response format")
                st.write(result)
        else:
            st.error(f"Error: {response.status_code}")
            try:
                st.write(response.json())
            except:
                st.write(response.text)
                
    except requests.exceptions.JSONDecodeError:
        st.error("❌ Invalid response from server. Response is not valid JSON.")
        st.write("Response text:", response.text)
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")





