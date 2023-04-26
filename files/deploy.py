import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon=":memo:", layout="wide")

model = joblib.load('./files/model.pkl')

st.title("Breast Cancer Diagnosis")

st.sidebar.header("User Input Parameters")

def user_input_features():
    radius_mean = st.sidebar.slider('radius_mean', 6.98, 28.1, 17.99)
    texture_mean = st.sidebar.slider('texture_mean', 9.71, 39.3, 10.38)
    perimeter_mean= st.sidebar.slider('perimeter_mean', 43.8, 189.0, 122.8)
    area_mean = st.sidebar.slider('area_mean', 144.0, 2500.0, 1001.0)
    smoothness_mean = st.sidebar.slider('Mean Smoothness', 0.05, 0.16, 0.1184)
    compactness_mean = st.sidebar.slider('Mean Compactness', 0.02, 0.35, 0.2776)
    concavity_mean = st.sidebar.slider('Mean Concavity', 0.0, 0.43, 0.3001)
    concave_points_mean = st.sidebar.slider('Mean Concave Points', 0.0, 0.2, 0.1471)
    symmetry_mean = st.sidebar.slider('Mean Symmetry', 0.11, 0.3, 0.2419)
    fractal_dimension_mean = st.sidebar.slider('Mean Fractal Dimension', 0.05, 0.1, 0.07871)
    radius_se = st.sidebar.slider('Radius Error', 0.11, 2.87, 1.095)
    texture_se = st.sidebar.slider('Texture Error', 0.36, 4.88, .9053)
    perimeter_se = st.sidebar.slider('Perimeter Error', 0.76, 22.0, 8.589)
    area_se = st.sidebar.slider('Area Error', 6.8, 542.0, 153.4)
    smoothness_se = st.sidebar.slider('Smoothness Error', 0.0, 0.03, 0.006399)
    compactness_se = st.sidebar.slider('Compactness Error', 0.0, 0.14, 0.04904)
    concavity_se = st.sidebar.slider('Concavity Error', 0.0, 0.4, 0.05373)
    concave_points_se = st.sidebar.slider('Concave Points Error', 0.0, 0.05, 0.01587)
    symmetry_se = st.sidebar.slider('Symmetry Error', 0.01, 0.08, 0.03003)
    fractal_dimension_se = st.sidebar.slider('Fractal Dimension Error', 0.0, 0.03, 0.006193)
    radius_worst = st.sidebar.slider('Worst Radius', 7.93, 36.0, 25.38)
    texture_worst = st.sidebar.slider('Worst Texture', 12.0, 49.5, 17.33)
    perimeter_worst = st.sidebar.slider('Worst Perimeter', 50.4, 251.0, 184.6)
    area_worst = st.sidebar.slider('Worst Area', 185.0, 4250.0, 2019.0)
    smoothness_worst = st.sidebar.slider('Worst Smoothness', 0.07, 0.22, 0.1622)
    compactness_worst = st.sidebar.slider('Worst Compactness', 0.03, 1.06, 0.6656)
    concavity_worst = st.sidebar.slider('Worst Concavity', 0.0, 1.25, 0.7119)
    concave_points_worst = st.sidebar.slider('Worst Concave Points', 0.0, 0.29, 0.2654)
    symmetry_worst = st.sidebar.slider('Worst Symmetry', 0.16, 0.66, 0.4601)
    fractal_dimension_worst = st.sidebar.slider('Worst Fractal Dimension', 0.06, 0.21, 0.1189)
    data = {'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'smoothness_mean': smoothness_mean,
            'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean,
            'concave_points_mean': concave_points_mean,
            'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean,
            'radius_se': radius_se,
            'texture_se': texture_se,
            'perimeter_se': perimeter_se,
            'area_se': area_se,
            'smoothness_se': smoothness_se,
            'compactness_se': compactness_se,
            'concavity_se': concavity_se,
            'concave_points_se': concave_points_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
            'radius_worst': radius_worst,
            'texture_worst': texture_worst,
            'perimeter_worst': perimeter_worst,
            'area_worst': area_worst,
            'smoothness_worst': smoothness_worst,
            'compactness_worst': compactness_worst,
            'concavity_worst': concavity_worst,
            'concave_points_worst': concave_points_worst,
            'symmetry_worst': symmetry_worst,
            'fractal_dimension_worst': fractal_dimension_worst
            }
    features = pd.DataFrame(data, index=[0])
    print(features)
    return features


def app():
	df = user_input_features()

	st.subheader('User Input parameters')
	st.write(df)
	if st.button('Predict Diagnosis'):
		result = model.predict(df)
		st.subheader('Diagnosis')
		if(result==1):
		   st.write('Malignant: CANCEROUS')
		else:
		   st.write('Benign: NON-CANCEROUS')
	   
if __name__ == '__main__':
    result = app()
