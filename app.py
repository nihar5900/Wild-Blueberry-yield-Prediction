import streamlit as st
import numpy as np
import pandas as pd
import joblib

model=joblib.load('model/model.pkl')

st.set_page_config(page_title="Blue Berry Yield Prediction App",
                   page_icon="🍇", layout="wide")



st.markdown("<h1 style='text-align: center;'>Blue Berry Yield Prediction App 🍇</h1>", unsafe_allow_html=True)
def main():
    with st.form('predictin_form'):
        st.subheader("Enter the below Features:")

        clonesize=st.number_input("Clone Size: ",12.00,38.00,format="%.2f")
        honeybee=st.number_input("Honeybee : ",0.24,0.55,format="%.2f")
        bumbles=st.number_input("Bumbles : ",0.11,0.38,format="%.2f")
        andrena=st.number_input("Andrena : ",0.25,0.75,format="%.2f")
        osmia=st.number_input("Osmia : ",0.05,0.75,format="%.2f")
        AvrageTRange=st.number_input("AvrageTRange : ",49.00,68.00,format="%.2f")
        AverageRainingDays=st.number_input("AverageRainingDays : ",0.06,0.56,format="%.2f")


        submit=st.form_submit_button("Predict")
    if submit:
        X=np.array([[clonesize,honeybee,bumbles,andrena,osmia,AvrageTRange,AverageRainingDays]])

        pred=model.predict(X)

        st.write(f"The predicted yield will be:  {pred[0]}")

if __name__=='__main__':
    main()