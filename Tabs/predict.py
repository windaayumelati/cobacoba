import streamlit as st
from web_functions import predict

def app(df, x, y):

    st.title("Halaman Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        long_hair_options = {0: 'No', 1: 'Yes'}
        long_hair = st.selectbox('Panjang Rambut', options=list(long_hair_options.keys()), format_func=lambda x: long_hair_options[x])

        forehead_width_cm = st.number_input('Input Lebar Dahi', min_value=11.4, max_value=15.5)
        forehead_height_cm = st.number_input('Input Panjang Dahi cp', min_value=5.1, max_value=7.1)
        nose_wide_options = {0: 'No', 1: 'Yes'}
        nose_wide = st.selectbox('Hidung Pesek', options=list(nose_wide_options.keys()), format_func=lambda x: nose_wide_options[x])

    with col2:
        nose_long_options = {0: 'No', 1: 'Yes'}
        nose_long = st.selectbox('Hidung Panjang', options=list(nose_long_options.keys()), format_func=lambda x: nose_long_options[x])
        
        lips_thin_options = {0: 'No', 1: 'Yes'}
        lips_thin = st.selectbox('Bibir Tipis', options=list(lips_thin_options.keys()), format_func=lambda x: lips_thin_options[x])

        distance_nose_to_lip_long_options = {0: 'No', 1: 'Yes'}
        distance_nose_to_lip_long = st.selectbox('BInput Jarak antara hidung dan panjang bibir', options=list(lips_thin_options.keys()), format_func=lambda x: distance_nose_to_lip_long_options[x])

    features = [long_hair, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long]


    # tombol prediksi
    if st.button("Prediksi Gender"):
        prediction, score = predict(x, y, features)
        
        gender_prediction = "Male" if prediction == 1 else "Female"
        st.info(f"Prediksi Gender: {gender_prediction}")
        st.info(f"Menggunakan KNeighborsClassifier(n_neighbors=3)")


      

  