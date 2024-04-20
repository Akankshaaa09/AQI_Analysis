import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import folium
import seaborn as sns
from streamlit_folium import folium_static

st.title('AQI Analysis')

user_input=st.text_input('Enter City')
df=pd.read_csv('city_day.csv')
df=pd.DataFrame(df)

#Describing Data
st.subheader('Data from 2015 - 2020')
city_data = df[df['City'] == user_input]
st.write(city_data)

#Visualizations

'''st.subheader("Average AQI In Cities")
fig=plt.figure(figsize=(30,10))
plt.bar(df['City'], df['AQI'])
plt.xlabel('City')
plt.ylabel('AQI')
st.pyplot(fig)
st.subheader("AQI Trends Over Time For City Entered By User")
fig=plt.figure(figsize=(12,6))
plt.plot(city_data['Date'], city_data['AQI'])
plt.xlabel("Date")
plt.ylabel("AQI")
st.pyplot(fig)'''

#Spatial Distribution
# Create a map centered around a specific location
st.subheader("Spatial Representation")
latitude_center = 20.5937
longitude_center = 78.9629
m = folium.Map(location=[latitude_center, longitude_center], zoom_start=5)

# Get unique cities from the dataset
unique_cities = df['City'].unique()

# Add markers for each unique city in the dataset
for city in unique_cities:
    city_data = df[df['City'] == city]
    # Get latitude and longitude of the city
    latitude = city_data['Latitude'].iloc[0] # Assuming latitude is in the dataset
    longitude = city_data['Longitude'].iloc[0] # Assuming longitude is in the dataset
    # Calculate average pollutant concentration for the city
    avg_aqi = city_data['AQI'].mean()  # You can choose any pollutant column for concentration
    def get_AQI_bucket(x):
      if x <= 50:
          return "Good"
      elif x <= 100:
          return "Satisfactory"
      elif x <= 200:
          return "Moderate"
      elif x <= 300:
          return "Poor"
      elif x <= 400:
          return "Very Poor"
      elif x > 400:
          return "Severe"
      else:
          return "N/A"
    # Add marker for the city
    folium.Marker(location=[latitude, longitude],
                  popup=f"City: {city}, Average AQI : {avg_aqi}, AQI Bucket : {get_AQI_bucket(avg_aqi)}", 
                  icon=folium.Icon(color='red')).add_to(m)

# Display the map on the Streamlit app
folium_static(m, width=800, height=600)

#Load model
dt_classifier = joblib.load('decision_tree_model.pkl')
st.subheader("Predict AQI")
pm25=st.number_input('Enter PM2.5 Value',step=0.0001)
pm10=st.number_input('Enter PM10 Value',step=0.0001)
so2=st.number_input('Enter SO2 Value',step=0.0001)
nox=st.number_input('Enter NOx Value',step=0.0001)
nh3=st.number_input('Enter NH3 Value',step=0.0001)
co=st.number_input('Enter CO Value',step=0.0001)
o3=st.number_input('Enter O3 Value',step=0.0001)
aqi=dt_classifier.predict([[pm25,pm10,so2,nox,nh3,co,o3]])
#st.text("The value is: {}".format(aqi))
st.write(aqi)
