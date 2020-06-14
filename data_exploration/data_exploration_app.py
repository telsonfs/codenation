import pandas as pd 
import numpy as np 
import streamlit as st


def open_dataset():

    path = st.file_uploader("Change your dataset", type = 'csv')
    try:
        dataset = pd.read_csv(path)
        return dataset
    except:
        print("Something went wrong when writing to the file")

def change_columns(df):
    selected_columns = st.multiselect('Select Columns', df.columns)
    return selected_columns
    
def main():
    st.title("Data Exploration")
    df = open_dataset()
    
    if df is not None:
        st.write(df.describe())
        
        selectbox = st.sidebar.selectbox('Select Columns', df.columns)
        columns = change_columns(df)
        st.write(df[columns].head(n=100))

        


if __name__== '__main__':
    main()