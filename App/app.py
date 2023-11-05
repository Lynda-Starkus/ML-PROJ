import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image('./logo.png')
    st.title("Projet ML")
    choice = st.radio("Navigation", ["Upload","Analyse des données","Modèlisation automatique", "Sauvegarder"])
    st.info("Application qui regroupe les modèles de classification et régression pour notre projet ML (Paris-Saclay M2 Datascale).")
if choice == "Upload":
    st.title("Charger le dataset")
    file = st.file_uploader("Charger le dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Analyse des données": 
    st.title("Etude exploratoire")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modèlisation automatique": 
    mdl = st.selectbox('Choisir un type de modèles : ',['Classification','Regression'])
    chosen_target = st.selectbox('Choisir la colonne cible : ', df.columns)
    if st.button('Exécuter'):
        if mdl == 'Classification':
            from pycaret.classification import setup, compare_models, pull, save_model 
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df.astype(str))
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
        elif mdl == 'Regression':
            from pycaret.regression import setup, compare_models, pull, save_model 
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df.astype(str))
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')


if choice == "Sauvegarder": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Sauvegarder le modèle', f, file_name="best_model.pkl")