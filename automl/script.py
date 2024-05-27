import streamlit as st
import pandas as pd
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import pycaret.classification as classification
import pycaret.regression as regression


with st.sidebar:
    st.image('icon.png')
    st.title('Automated ML')
    choice = st.radio("Navigation", ['Upload', 'Data Analysis', 'Regression Modelling','Classification Modelling', 'Regressor Testing', 'Classifier Testing'])
    st.info('An AI-driven AutoML web app that automates the process of building, training, and deploying machine learning models, making data science accessible to everyone.')

if os.path.exists('original_data.csv'):
    df = pd.read_csv('original_data.csv', index_col=None)
if choice == 'Upload':
    file = st.file_uploader('Upload your Data here')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('original_data.csv', index =None)
        st.dataframe(df)

if choice == 'Data Analysis':
    st.title('Exploratiory Data Analysis')
    report = ProfileReport(df)
    st_profile_report(report)

if choice == 'Regression Modelling':
    st.title('Generation of Regression model')
    target = st.selectbox('Select Target Parameter', df.columns)
    if st.button('Train Model'):
            regression.setup(df, target=target)
            setup_df = regression.pull()
            st.info('Model Experimentation Setiings')
            st.dataframe(setup_df)
            best_model = regression.compare_models()
            compare_df = regression.pull()
            st.info('Generated Regressor')
            st.dataframe(compare_df)
            best_model
            regression.save_model(best_model, 'best_regressor')
            st.info('You Can Download the generated model from here')
            with open('best_regressor.pkl', 'rb') as f:
                st.download_button('Download the Model', f, 'trained_regressor.pkl')
if choice == 'Classification Modelling':
    st.title('Generation of Classification model')
    target = st.selectbox('Select Target Parameter', df.columns)
    if st.button('Train Model'):
            classification.setup(df, target=target)
            setup_df = classification.pull()
            st.info('Model Experimentation Setiings')
            st.dataframe(setup_df)
            best_model = classification.compare_models()
            compare_df = classification.pull()
            st.info('Generated Classifier')
            st.dataframe(compare_df)
            best_model
            classification.save_model(best_model, 'best_classifier')
            st.info('You Can Download the generated model from here')
            with open('best_classifier.pkl', 'rb') as f:
                st.download_button('Download the Model', f, 'trained_classifier.pkl')
if choice == 'Regressor Testing':
    st.title('Model Testing')
    st.info('Upload your test dataset')
    test = st.file_uploader('Upload Test csv file')
    if test is not None:  
        df2 = pd.read_csv(test)
        st.info('This is your test dataset')
        st.dataframe(df2)
        
        pipeline = regression.load_model('trained_regressor')
        st.info('This is the dataframe with the prediction labels')
        st.dataframe(regression.predict_model(pipeline, df2))
    else:
        st.warning('Please upload a test dataset before proceeding.') 

if choice == 'Classifier Testing':
    st.title('Model Testing')
    st.info('Upload your test dataset')
    test = st.file_uploader('Upload Test csv file')
    if test is not None:  
        df2 = pd.read_csv(test)
        st.info('This is your test dataset')
        st.dataframe(df2)
        
        pipeline = classification.load_model('trained_classifier')
        st.info('This is the dataframe with the prediction labels')
        st.dataframe(classification.predict_model(pipeline, df2))
    else:
        st.warning('Please upload a test dataset before proceeding.')  

