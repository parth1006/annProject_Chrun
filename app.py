import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
 
model=tf.keras.models.load_model('model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
onehot= pickle.load(open('oheencoder.pkl', 'rb'))
label=pickle.load(open('labelencoder.pkl','rb'))

st.title('Customer churn prediction')
st.write('This is a simple web app to predict customer churn using a neural network model')
geography=st.selectbox('Select the country',onehot.categories_[0])
gender=st.selectbox('Gender', label.classes_)
age=st.slider('Age',18,30)
balance=st.number_input('Balance')
credit_input=st.number_input('Credit score')
estimatedsal=st.number_input('Estimated salary')
tenure =st.slider("Tenure",0,10)
no_of_products=st.slider("Number of prodcuts",1,5)
has_Cr_credit=st.selectbox('Has credit card', [1, 0])
is_Active=st.selectbox('Is active member', [1, 0])

input_data=pd.DataFrame({
    'CreditScore':[credit_input],
    'Gender': [label.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [no_of_products],
    'HasCrCard': [has_Cr_credit],
    'IsActiveMember': [is_Active],
    'EstimatedSalary': [estimatedsal],
})

geo_encoded=onehot.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]


st.write('Prediction Probability:',prediction_proba)
if prediction_proba>0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn  ')