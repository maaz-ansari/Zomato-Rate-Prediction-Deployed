import streamlit as st
import pickle
import numpy as np
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1SBJDhpzTpf4Y2d8CVs9OdjWPufUN492j',
                                     dest_path='./Zom_cleaned.csv',
                                     unzip=False)
gdd.download_file_from_google_drive(file_id='1VQ5jiCwH337mb5t91qi8hJwSILp3Qo54',
                                     dest_path='./zom_mod.pkl',
                                     unzip=False)

df=pd.read_csv('Zom_cleaned.csv')
model=pickle.load(open("zom_mod.pkl",'rb'))

df=df.drop(['url','address','name','phone','dish_liked','reviews_list','listed_in(city)' ],1)
df=df.dropna()

loc_list = pd.get_dummies(df[['location', 'listed_in(type)']], prefix='', prefix_sep='')

df['rest_type'] = df['rest_type'].str.replace(' ','')
rest_dummy=df['rest_type'].str.get_dummies(sep=',')

df['cuisines'] = df['cuisines'].str.replace(' ','')
cuisines_dummies=df['cuisines'].str.get_dummies(sep=',')

def main():
    st.subheader('Author: Maaz Ansari')
    st.title("Zomato Rate Prediction")
    
    votes= st.text_input("Votes",800,key='Vote')
    approx_cost = st.text_input("Approx cost",900, key='Cost')

    online_order = st.selectbox('Online Order',('Yes', 'No'), key='online')

    book_table = st.selectbox('Book Table',('Yes', 'No'), key='book')

    location = st.selectbox('Location', (df['location'].unique()), key='loc')

    cuisines = str(st.multiselect('Cuisines', (cuisines_dummies.columns.sort_values()), key='cui'))

    rest_type = str(st.multiselect('Restaurant Type', (rest_dummy.columns.sort_values()), key='rt'))

    listed_in = st.selectbox('listed_in(type)', (df['listed_in(type)'].unique()), key='lt')


    dt = pd.DataFrame([[votes,approx_cost, online_order, book_table, location, cuisines, rest_type, listed_in]], columns=['votes', 'approx_cost', 'online_order', 'book_table', 'location', 'cuisines', 'rest_type','listed_in']) 
    df1 = dt[['votes', 'approx_cost']]

    df2 = dt[['online_order','book_table']]
    df2['online_order'] = df2['online_order'].replace('Yes', 1)
    df2['online_order'] = df2['online_order'].replace('No', 0)
    df2['book_table'] = df2['book_table'].replace('Yes', 1)
    df2['book_table'] = df2['book_table'].replace('No', 0)

    df3 = dt[['location', 'listed_in']]
    l1 = pd.get_dummies(df3)
    df3 = l1.reindex(columns = loc_list.columns, fill_value=0)

    df4 = dt[['rest_type']]
    df4['rest_type'] = df4['rest_type'].str.replace(' ','')
    r1=df4['rest_type'].str.get_dummies(sep=',')
    df4 = r1.reindex(columns = rest_dummy.columns, fill_value=0)

    df5 = dt[['cuisines']]
    df5['cuisines'] = df5['cuisines'].str.replace(' ','')
    c1 = df5['cuisines'].str.get_dummies(sep=',')
    df5 = c1.reindex(columns = cuisines_dummies.columns, fill_value=0)

    final_df = pd.concat([df1,df2,df3,df4,df5],axis=1)
    
    output=""
    if st.button("Predict"):
        output=model.predict(final_df)
        output=round(output[0],2)
        st.success('The Restaurant Rating is {}'.format(output))

if __name__=='__main__':
     main()

