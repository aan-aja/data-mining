import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns 
import pickle 

#import model 
rf = pickle.load(open('RF.pkl','rb'))

#load dataset
data = pd.read_csv('/content/drive/MyDrive/Cirhossis Dataset.csv')

st.title('Aplikasi Cirhossis')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Cirhossis Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Random Forest','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Cirhossis</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDA'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('Stage',axis=1)
y = data['Stage']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    Age = st.sidebar.number_input('Enter your age: ',20 , 60)
    Sex  = st.sidebar.selectbox('Sex',(0,1,2))
    Ascites = st.sidebar.selectbox('Asites',(0,1,2))
    Hepatomegaly = st.sidebar.selectbox('Hepatomegali', (0,1,2))
    Spiders = st.sidebar.selectbox('Spider', (0,1,2))
    Edema = st.sidebar.selectbox('Edema',(0,1,2))
    Bilirubin = st.sidebar.number_input('Bilirubin: ', 0.4, 22.0)
    Cholesterol = st.sidebar.number_input('Kolesterol: ',0, 1775)
    Albumin = st.sidebar.number_input('Albumin: ', 0.0, 4.0)
    Copper = st.sidebar.number_input('Copper ', 0,588)
    Alk_Phos = st.sidebar.number_input('hAlkPhos: ',0,11552)
    SGOT = st.sidebar.number_input('SGOT: ',0, 338)
    Tryglicerides = st.sidebar.selectbox('Tryglicerides: ', (0,1))
    Platelets = st.sidebar.selectbox('Platelets ',(0,1))
    Prothrombin = st.sidebar.selectbox('Prothrombin: ',(0,1))
    Stage = st.sidebar.selectbox('Stage: ', (0,1))
    
    user_report_data = {
        'Umur':Age,
        'sex':Sex,
        'Asites':Ascites,
        'Hepatomegali':Hepatomegaly,
        'Spider':Spiders,
        'Edema':Edema,
        'Bilirubin':Bilirubin,
        'Kolesterol':Cholesterol,
        'Albumin':Albumin,
        'Copper':Copper,
        'AlkPhos':Alk_Phos,
        'SGOT':SGOT,
        'Tryglicerides':Tryglicerides,
        'Platelets':Platelets,
        'Prothrombin':Prothrombin,
        'Stage':Stage
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasien
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = rf.predict(user_data)
rf_score = accuracy_score(y_test,rf.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena cirhossis'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(rf_score*100)+'%')
