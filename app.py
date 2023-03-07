import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import streamlit.components.v1 as components
from PIL import Image

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://github.com/llorenc-fer/Allianz/blob/main/Sin%20t%C3%ADtulo.png?raw=true");
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_url()


regressionscores = pd.read_csv('https://github.com/llorenc-fer/Allianz/blob/main/regressionscores.csv', sep='delimiter', header=None)
nulls = pd.read_csv('https://github.com/llorenc-fer/Allianz/blob/main/nulls.csv',sep='delimiter', header=None)
paid_record = pd.read_csv('https://github.com/llorenc-fer/Allianz/blob/main/paid_record.csv',sep='delimiter', header=None)
paid_recordshape = paid_record.shape
main = pd.read_csv('https://github.com/llorenc-fer/Allianz/blob/main/mainsample.csv',sep='delimiter', header=None)
mainshape = main.shape
address = pd.read_csv('https://github.com/llorenc-fer/Allianz/blob/main/address.csv',sep='delimiter', header=None)
addressshape = address.shape
coldic = Image.open('https://raw.githubusercontent.com/llorenc-fer/Allianz/blob/main/Explicacion_columnas.png')
histbins = Image.open('https://raw.githubusercontent.com/llorenc-fer/Allianz/blob/main/histogram%20columns.png')
corrmap = Image.open('https://raw.githubusercontent.com/llorenc-fer/Allianz/blob/main/corrmap.png')
importances1 = pd.read_csv('https://github.com/llorenc-fer/Allianz/blob/main/importances1.csv',sep='delimiter', header=None)
importances2 = pd.read_csv('https://github.com/llorenc-fer/Allianz/blob/main/importances2.csv',sep='delimiter', header=None)
importances3 = pd.read_csv('https://github.com/llorenc-fer/Allianz/blob/main/importances3.csv',sep='delimiter', header=None)

st.title("Allianz Data Talent Datathon 2023")
st.markdown("This application is part of the 1st edition of the Allianz Data Talent Program Datathon (2023)")



#-----sidebar------------------------------------------------------------------------------------------
st.set_option('deprecation.showPyplotGlobalUse', False) #para que no muestre warnings de versiones desfasadas
st.sidebar.title('Menu')

if st.sidebar.button('Introduction'):
    st.subheader("Instructions provided by Allianz & NTT Data")
    st.markdown("""

Firstly, you will be provided with three databases:\n
- The first one contains historical information about payment (or non-payment) of insurance premium instalments for a set of IDs, contracted with an insurance company. This information is presented in historical series ranging from 1 to 60 months. \n
- The second table contains qualitative information for a series of IDs. \n
- The third contains the complete address, zip code, and state within the United States.\n

**Objective**: Based on the previously mentioned data, **can we determine which new customers are eligible for a free premium instalment plan?**

You should assume that the same qualitative information present in the challenge's database will be available for these new customers.
""")
    
if st.sidebar.button('Datasets Overview'):
    st.subheader("Main Dataset")
    st.dataframe(main)
    st.write('Shape: (438757, 19)')
    st.subheader("Paid Record Dataset")
    st.dataframe(paid_record)
    st.write("Shape: (1048575, 3)")
    st.subheader("Address dataset")
    st.dataframe(address)
    st.write('Shape: (438757, 2)')
    st.subheader('Variable Dictionary')
    st.image(coldic, caption='Column Dictionary provided by Allianz and NTT Data')
#---------------------------------------------------------------------------------------------------------------

if st.sidebar.button('Preprocessing'):
    st.markdown(""" 
    - ID columns renamed and format-modified for all datasets, Main dataset and Address dataset merged on ID column.
    """)
    st.subheader("Null Data Treatment")
    st.dataframe(nulls)
    st.markdown("""

- Dropped 'OCCUPATION_TYPE' as it is unusable.
- Dropped 'Letter' as it doesn't provide valuable information.
- Checked for outliers in numerical columns and imputated them before repairing nulls (so that mean doesn't affect the reparating)
- Repared 'CNT_Children' nulls with mode
- Repared nulls after outliers have been treated.
- Converted 'DAYS_BIRTH_CLEAN' into datetime format to calculate Age into a new column called 'Age'.
- Mapped binary categorical values with numerical values.
""")
    st.write("Column 'CNT_CHILDREN' before reparing outliers")
    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/childrenoutliersbefore.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)
    st.write("Column 'CNT_CHILDREN' after reparing outliers")
    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/childrenoutliersafter.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)
    st.write("Column 'AMT_INCOME_TOTAL' before reparing outliers")
    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/incomeoutliersbefore.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)
    st.write("Column 'AMT_INCOME_TOTAL' after reparing outliers")
    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/incomeoutliersafter.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)
    st.subheader("Feature Engineering")
    st.markdown("""
In this part we studied the features of our dataset to:
- Create our target 
- Delete unnecessary columns 
- Determine columns not correlated with the target 
- Delete redundant columns and create new columns

""")
    st.subheader('Correlation Map')
    st.image(corrmap)
    st.markdown("""
- Dropped column 'DAYS_EMPLOYED' (redundant)
- Dropped column 'FLAG_OWN_REALTY' (uncorrelated to target)
- Dropped column 'CNT_FAM_MEMBERS' (redundant)
- Dropped column 'ID' (uncorrelated to target)


**Feature engineering** for Paid Records Dataframe: 
- Dropped all rows where status is X since it doesn't add up any new information
- Created a column that contains the amount of total months per customer
- Created a column that contains the amount of months paid per customer
- Created a column that contains the ratio of months paid to total months
- Dropped all repeated ID rows.
- Merged Paid Records Dataframe to our main Dataframe.
- Create a 'target' column based on ratio

""")
    st.subheader("Target Threshold")
    st.markdown("""The target category (Not eligible/Eligible for a free premium instalment plan) is decided by a threshold set in the ratio score.
    We have chosen a threshold of 0.8, but one of the strong points of this system is that it allows to easily change the threshold, 
    hence giving a lot of flexibility.""")
    st.subheader('Column Histogram')
    st.markdown("After preprocessing the columns, here's what their distribution looks like")
    st.image(histbins)

#---------------------------------------------------------------------------------------------------------------

if st.sidebar.button('Data Visualisation'):
    st.subheader("Data Visualisation")

    st.write("Population overview")
    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/pyramid.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)

    st.write("Income by gender")
    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/Incomebygender.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)

    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/numberofchildren.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)

    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/realtyownership.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)

    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/nameeducationtype.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)

    html = open("https://github.com/llorenc-fer/Allianz/blob/main/Graphs/owncar.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)

#-----------------------------------------------------------------------------------------
if st.sidebar.button('Feature Importance'):
    st.subheader("Feature Importance Tests")
    st.markdown("Before starting our Machine Learning Models, we performed three different Feature Importance Tests to make sure that our variables were important to the target. This is how it went:")
    st.subheader("Random Forest Classification Feature Importance")
    st.dataframe(importances1)
    st.markdown("""
- We can see that 'age' and 'AMT_INCOME_TOTAL' are important features for predicting the target variable, with 'age' having an importance of 0.3227 and 'AMT_INCOME_TOTAL' having an importance of 0.2985.
- Other features such as 'CNT_CHILDREN', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', and 'NAME_HOUSING_TYPE' also have some level of importance, but they are less important than the top two features.
- Features such as 'CODE_GENDER' and 'FLAG_OWN_CAR' have relatively low importance values, indicating that they may not be useful in predicting the target variable
                """)

    st.subheader("Correlation-based Feature Selection")
    st.dataframe(importances2)
    st.markdown("""
- 'NAME_EDUCATION_TYPE' has the strongest correlation with the target variable. 
- The features 'NAME_FAMILY_STATUS' and 'FLAG_OWN_CAR' also have a relatively strong correlation with the target variable.
- The features 'FLAG_WORK_PHONE' and 'FLAG_PHONE' have a moderate correlation with the target variable. 
- 'AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CODE_GENDER', and 'NAME_HOUSING_TYPE' have a weaker correlation with the target variable. The feature 'CODE_GENDER' also shows a weak correlation.    
    
    """)

    st.subheader("Permutation Importance")
    st.dataframe(importances3)
    st.markdown("""
- Top 5 features positively correlated with the target variable are: AMT_INCOME_TOTAL, age, NAME_FAMILY_STATUS, CNT_CHILDREN, and FLAG_PHONE.
- Top 5 features negatively correlated with the target variable are: NAME_EDUCATION_TYPE, NAME_INCOME_TYPE, FLAG_WORK_PHONE, CODE_GENDER, and NAME_HOUSING_TYPE.
- The magnitude of correlation coefficients ranges from 0.246 for NAME_INCOME_TYPE to 0.6743 for CODE_GENDER among positively correlated features, and from -0.056 for NAME_HOUSING_TYPE to -0.002 for NAME_EDUCATION_TYPE and NAME_FAMILY_STATUS among negatively correlated features.
    
    """)

    st.subheader("Conclusions")
    st.markdown("""
- 'age', 'AMT_INCOME_TOTAL', and 'CNT_CHILDREN' are consistently among the top features across **all three methods**. This indicates that these features are likely to have a strong influence on the target variable and should be considered important in predicting the target.

- Additionally, we see that 'NAME_EDUCATION_TYPE' and 'NAME_FAMILY_STATUS' are among the top features selected by both **correlation-based feature selection** and **random forest classifier feature importance**. This suggests that these features may be especially important in predicting the target and should be given additional consideration.

- On the other hand, the **permutation importance method** yields somewhat different results, with 'NAME_INCOME_TYPE', 'FLAG_WORK_PHONE', and 'NAME_HOUSING_TYPE' being the top three features. 
    
    """)
#--------------------------------------------------------------------------------------------------------------
st.sidebar.write('Machine Learning')
if st.sidebar.button('Classification'):
    st.subheader("-")

#--------------------------------------------------------------------------------------------------------------
if st.sidebar.button('Regression'):
    st.markdown("""
    Since we had already created the column ratio, we have tried several Machine Learning Models to check if predicting the ratio score with a regression could be more useful than predicting the category classification of the target column. \n
    Here are the results:""")
    st.dataframe(regressionscores)
