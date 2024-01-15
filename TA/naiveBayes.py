import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Membuat halaman Streamlit
st.title("Heart Attack Prediction App")

# Menampilkan informasi aplikasi
st.subheader("Application Info")
st.write(
    "This is a simple Heart Attack Prediction application using the Naive Bayes algorithm. "
    "The dataset contains various attributes related to heart health, and the model is trained to predict the likelihood of a heart attack."
)

# Membaca dataset dari file CSV
df = pd.read_csv('heart_attack_ds.csv')

# Menampilkan seluruh dataset di bagian utama aplikasi
st.subheader("Full Dataset Preview")
st.write(df)

# Mengkonversi kolom 'sex' menjadi numerik (jika diperlukan)
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

# Mengkonversi kolom 'cp' menjadi numerik
cp_mapping = {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3}
df['cp'] = df['cp'].map(cp_mapping)

# Memisahkan fitur dan label
X = df.drop('output', axis=1)
y = df['output']

# Membagi dataset menjadi training (80%), validation (10%), dan test (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Inisialisasi dan melatih model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Mengukur kinerja pada training set
accuracy_train = accuracy_score(y_train, model.predict(X_train))

# Melakukan prediksi pada validation set
y_val_pred = model.predict(X_val)

# Mengukur kinerja pada validation set
accuracy_val = accuracy_score(y_val, y_val_pred)

# Melakukan prediksi pada test set
y_test_pred = model.predict(X_test)

# Mengukur kinerja pada test set
accuracy_test = accuracy_score(y_test, y_test_pred)

# Menampilkan akurasi pada training set
st.subheader("Model Accuracy")
st.write(f"Accuracy on training set: {accuracy_train:.2f}")
st.write(f'Accuracy on validation set: {accuracy_val:.2f}')
st.write(f'Accuracy on test set: {accuracy_test:.2f}')

# Sidebar untuk input data
st.sidebar.header("Input Data")

# Menerima input dari pengguna
age = st.sidebar.number_input("Age", min_value=29, max_value=77, value=50)
sex = st.sidebar.radio("Sex", ['male', 'female'])
cp = st.sidebar.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=94, max_value=200, value=120)
chol = st.sidebar.number_input("Cholesterol", min_value=126, max_value=564, value=240)
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=71, max_value=202, value=150)

# Tombol "Run Prediction"
if st.sidebar.button("Run Prediction"):
    # Menyiapkan data input pengguna
    user_input = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'male' else 0],
        'cp': [cp_mapping[cp]],
        'trestbps': [trestbps],
        'chol': [chol],
        'thalach': [thalach],
    })

    # Melakukan prediksi untuk data input pengguna
    prediction = model.predict(user_input)

    # Menampilkan hasil prediksi
    st.subheader("Prediction Result:")
    st.write("The model predicts:", "Heart Attack" if prediction[0] == 1 else "No Heart Attack")

