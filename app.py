import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from math import sqrt, pi, exp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler  # Import RandomUndersampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

header_html = """
    <style>
        header {
            background-color: #4CAF50;  /* Warna background */
            padding: 10px 0;  /* Menambahkan padding untuk jarak atas dan bawah */
            text-align: center;  /* Menyusun teks di tengah */
            margin: 0;  /* Menghilangkan margin */
        }
        header h1 {
            color: white;  /* Warna teks */
            margin: 0;  /* Menghilangkan margin untuk header */
        }
    </style>
    <header>
        <h1>Prediksi Status Gizi Tumbuh Kembang Anak 👶</h1>
    </header>
"""

# Menampilkan HTML di Streamlit
st.markdown(header_html, unsafe_allow_html=True)

# Konten utama aplikasi
# st.title("Ini adalah contoh aplikasi Streamlit dengan header yang disesuaikan")
# st.write("Di bawah header ini, kamu bisa menambahkan berbagai elemen dan interaksi.")

# === Fungsi Gaussian Naive Bayes ===
def gaussian(x, mean, std):
    eps = 1e-6
    return (1 / (sqrt(2 * pi) * (std + eps))) * exp(-((x - mean) ** 2) / (2 * (std + eps) ** 2))

# === Fungsi prediksi satu baris ===
def predict_row(row, priors, mean_std, features, classes):
    probs = dict()
    for cls in classes:
        prob = priors[cls]
        for col in features:
            mean = mean_std[cls]['mean'][col]
            std = mean_std[cls]['std'][col]
            prob *= gaussian(row[col], mean, std)
        probs[cls] = prob
    return max(probs, key=probs.get)

# === Aplikasi Streamlit ===
# Konfigurasi halaman
# st.set_page_config(page_title="Analisis Tumbuh Kembang Anak", page_icon="👶", layout="wide")

# Judul aplikasi
# st.title('Analisis Dataset Tumbuh Kembang Anak')
st.markdown("---")

# Membuat tab
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Preprocessing Data", "Split & Sampling", "Modeling", "Prediksi Data"])

# Tab 1 - Upload Dataset
with tab1:
    st.markdown("## 👶 Prediksi Status Gizi Tumbuh Kembang Anak")
    st.markdown("""
    Aplikasi ini dirancang untuk membantu memprediksi **status gizi anak** berdasarkan data antropometri
    seperti **jenis kelamin, usia, berat badan, tinggi badan, dan lingkar lengan atas (LiLA)**.

    🔍 **Tujuan Aplikasi**:
    - Menerapkan algoritma *Naïve Bayes* yang diperkuat dengan *AdaBoost* untuk klasifikasi status gizi.
    - Mengatasi ketidakseimbangan data menggunakan teknik *SMOTE* dan *Random Undersampling*.
    - Memberikan visualisasi hasil evaluasi model dan memungkinkan pengguna mencoba prediksi data baru.

    🚀 **Cara Menggunakan**:
    1. Masuk ke tab **Preprocessing Data** untuk melihat pembersihan dan penyiapan data.
    2. Gunakan tab **Split & Sampling** untuk membagi dan menyeimbangkan data.
    3. Jalankan model di tab **Modeling** dan lihat hasil evaluasinya.
    4. Coba prediksi data baru secara interaktif di tab **Prediksi Data**.

    📊 Dataset yang digunakan adalah data stunting dari Kaggle, berisi 1.288 data anak.

    """)
    st.success("Silakan lanjutkan ke tab berikutnya untuk memulai analisis.")

# Tab 2 - Preprocessing Data
with tab2:
    dataset_file = "Baru (1).csv"
    if dataset_file is not None:
        st.subheader("1. Load Dataset")
        dataset = pd.read_csv(dataset_file, encoding='ISO-8859-1')
        
        st.write(f"Total Record: {dataset.shape[0]}")
        st.write(f"Total Atribut: {dataset.shape[1]}")
        st.dataframe(dataset.head())
        
        
        st.subheader("2. Cleaning Data")
        drop_cols = ['Nama', 'Tgl Lahir', 'Nama Ortu', 'Desa/Kel', 'RT', 'RW',
                     'ZS BB/U', 'ZS TB/U', 'ZS BB/TB']
        dataset.drop(columns=drop_cols, inplace=True)

        st.write("Dataset Setelah Cleaning Data:")
        st.dataframe(dataset.head())
        
        st.subheader("3. Penanganan Data Duplikat")
        duplikat_per_kolom = {col: dataset[col].duplicated().sum() for col in dataset.columns}
        duplikat_saja = {col: jumlah for col, jumlah in duplikat_per_kolom.items() if jumlah > 0}
        df_duplikat = pd.DataFrame.from_dict(duplikat_saja, orient='index', columns=['Jumlah Duplikat'])
        st.write("Jumlah Data Duplikat per Kolom:")
        st.write(df_duplikat)
        st.write("Total Data Duplikat:")
        st.write(f"Total Data Duplikat: {dataset.duplicated().sum()}")  

        # Menghapus data duplikat
        dataset = dataset.drop_duplicates()

        # Menghapus baris yang memiliki nilai 0 di semua kolom & NaN selain LiLA
        dataset = dataset[(dataset != 0).all(axis=1)].dropna(subset=['Usia', 'Berat', 'Tinggi'])
        st.write("Dataset Setelah Pembersihan:")
        st.write(f"Total Record: {dataset.shape[0]}")
        st.dataframe(dataset.head())

        # Penanganan Missing Value
        st.subheader("4. Penanganan Missing Value")
        missing_values = dataset.isnull().sum()
        st.write("Jumlah missing value per kolom:")
        st.write(missing_values)
        st.write("Apakah ada missing value? ", dataset.isnull().values.any())

        st.write("Penanganan Missing Value (imputasi data)")
        # Imputasi kolom LiLA dengan modus
        knn_imputer = KNNImputer(n_neighbors=5)
        dataset[['LiLA']] = knn_imputer.fit_transform(dataset[['LiLA']])

        st.write("Setelah Imputasi:")
        st.dataframe(dataset.head())
        
        st.subheader("5. Penghapusan Satuan Kata")
        # Menghapus string ' bulan' pada kolom Usia
        dataset['Usia'] = dataset['Usia'].str.replace(' bulan', '')

        st.write("Setelah Penghapusan Satuan Kata:")
        st.dataframe(dataset[['Usia']].head())
        
        st.subheader("6. Konversi Tipe Data")
        dataset['Usia'] = dataset['Usia'].astype(int)
        st.write("Dataset Akhir Siap Digunakan:")
        st.dataframe(dataset.head())

        st.subheader("7. Encoding Data")
        # One-Hot Encoding untuk JK (L=0, P=1)
        # Pastikan hanya nilai yang valid
        dataset['JK'] = pd.get_dummies(dataset['JK'], drop_first=True, dtype=int)

        st.write("Setelah Encoding JK:")
        st.dataframe(dataset[['JK']].head())

        # Label Encoding untuk Status Gizi
        le_status = LabelEncoder()
        dataset['Status Gizi'] = le_status.fit_transform(dataset['Status Gizi'])
        classes = le_status.classes_
        encoded_values = le_status.transform(classes)
        classes_df = pd.DataFrame({'Status Gizi': classes, 'Hasil Encoding': encoded_values})

        # Info ke user
        st.info("Kolom 'JK': Laki-laki = 0, Perempuan = 1.\nLabel 'Status Gizi' telah diubah ke format numerik.")

        # Tampilkan hasil
        st.write("### Dataset Setelah Encoding:")
        st.dataframe(dataset.head())

        st.write("### Mapping Label Status Gizi:")
        st.dataframe(classes_df)

        st.success("Pembersihan data selesai!")

# Tab 3 - Split Data & Sampling
with tab3:
    if dataset_file is not None:
        st.subheader("Split Data & Sampling")

        # Pilih rasio split data
        split_option = st.selectbox("Pilih Rasio Split Data", ["90:10", "80:20", "70:30"])
        test_size = {"90:10": 0.1, "80:20": 0.2, "70:30": 0.3}[split_option]

        X = dataset[['JK', 'Usia', 'Berat', 'Tinggi', 'LiLA']]
        y = dataset['Status Gizi']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.write("Distribusi Kelas Sebelum Sampling:", dict(Counter(y_train)))
        fig1, ax1 = plt.subplots()
        sns.countplot(x=y_train, ax=ax1)
        ax1.set_title("Distribusi Kelas Sebelum Sampling")
        st.pyplot(fig1)

        # Pilihan metode sampling
        sampling_method = st.radio("Pilih Metode Sampling", ["SMOTE (Over Sampling)", "Random Undersampling"])

        if sampling_method == "SMOTE (Over Sampling)":
            # SMOTE
            sm = SMOTE(random_state=42)
            X_train_over, y_train_over = sm.fit_resample(X_train, y_train)
            st.write("Distribusi Kelas Setelah SMOTE:", dict(Counter(y_train_over)))
            fig2, ax2 = plt.subplots()
            sns.countplot(x=y_train_over, ax=ax2)
            ax2.set_title("Distribusi Kelas Setelah SMOTE (Over Sampling)")
            st.pyplot(fig2)
            sampling_type = "Over Sampling"
            X_train_resampled, y_train_resampled = X_train_over, y_train_over
        else:
            # Random Undersampling
            rus = RandomUnderSampler(random_state=42)
            X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
            st.write("Distribusi Kelas Setelah Random Undersampling:", dict(Counter(y_train_under)))
            fig3, ax3 = plt.subplots()
            sns.countplot(x=y_train_under, ax=ax3)
            ax3.set_title("Distribusi Kelas Setelah Random Undersampling (Under Sampling)")
            st.pyplot(fig3)
            sampling_type = "Under Sampling"
            X_train_resampled, y_train_resampled = X_train_under, y_train_under

        st.success(f"Split dan {sampling_type} selesai!")

# Tab 4 - Modeling
with tab4:
    if dataset_file is not None:
        st.subheader("Modeling: Naïve Bayes + AdaBoost")

        # Naive Bayes Manual
        classes = np.unique(y_train_resampled)
        features = X_train_resampled.columns
        priors = {}
        mean_std = {}

        for cls in classes:
            X_cls = X_train_resampled[y_train_resampled == cls]
            priors[cls] = len(X_cls) / len(X_train_resampled)
            mean_std[cls] = {
                'mean': X_cls.mean(),
                'std': X_cls.std()
            }

        y_pred_manual = X_test.apply(lambda row: predict_row(row, priors, mean_std, features, classes), axis=1)

        akurasi = accuracy_score(y_test, y_pred_manual)
        st.subheader("Hasil Evaluasi Naive Bayes")
        st.write("Akurasi:", round(akurasi * 100, 2), "%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_manual))

        cm = confusion_matrix(y_test, y_pred_manual, labels=classes)
        fig3, ax3 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes, ax=ax3)
        ax3.set_title("Confusion Matrix - Naive Bayes Manual")
        ax3.set_xlabel("Prediksi")
        ax3.set_ylabel("Aktual")
        st.pyplot(fig3)

        report = classification_report(y_test, y_pred_manual, output_dict=True)
        

        # Naive Bayes dan AdaBoost
        n_estimators = st.slider("Number of AdaBoost estimators", 10, 100, 50, 10)
        if st.button("Jalankan Naïve Bayes + AdaBoost"):
            st.subheader(f"AdaBoost dengan {n_estimators} Estimator")
            # Inisialisasi dan latih model
            adaboost = AdaBoostClassifier(estimator=GaussianNB(), n_estimators=n_estimators, random_state=42)
            adaboost.fit(X_train_resampled, y_train_resampled)
            st.session_state.adaboost = adaboost
            adaboost_pred_test = adaboost.predict(X_test)

            # Evaluasi
            acc = accuracy_score(y_test, adaboost_pred_test)
            cm = confusion_matrix(y_test, adaboost_pred_test)
            report = classification_report(y_test, adaboost_pred_test)
            st.write(f"Akurasi: {round(acc*100, 2)}%")
            
            # Tampilkan Classification Report
            st.write("Classification Report:")
            st.text(report)
            
            # === Tampilkan detail iterasi boosting ===
            st.subheader("Detail Iterasi AdaBoost")
            est_weights = adaboost.estimator_weights_
            est_errors = adaboost.estimator_errors_
            n_actual_iters = len(est_weights)
            
            st.write(f"Jumlah Iterasi (Estimator Aktif): {n_actual_iters}")
            
            for i in range(n_actual_iters):
                st.markdown(f"""
                **Iterasi ke-{i+1}:**
                - Bobot (alpha): `{round(est_weights[i], 4)}`
                - Error: `{round(est_errors[i], 4)}`
                """)

            # Heatmap
            fig4, ax4 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes, ax=ax4)
            ax4.set_title(f"Confusion Matrix - AdaBoost ({n_estimators} Estimator)")
            ax4.set_xlabel("Prediksi")
            ax4.set_ylabel("Aktual")
            st.pyplot(fig4)
            st.success("Modeling selesai!")

# Tab 5 - Coba Data
with tab5:
    st.subheader("Prediksi Data")

    if dataset_file is None:
        st.warning("Silakan upload dataset terlebih dahulu.")
    elif 'adaboost' not in st.session_state:
        st.warning("Model belum dibuat. Jalankan modeling di tab sebelumnya.")
    else:
        # Masukkan data baru untuk prediksi
        st.write("Masukkan data baru sesuai dengan format berikut:")
        input_data = {}
        input_data['JK'] = st.radio("Jenis Kelamin", options=["Laki-laki", "Perempuan"])
        input_data['Usia'] = st.number_input("Usia (bulan)", min_value=0, max_value=100, value=12)
        input_data['Berat'] = st.number_input("Berat Badan (kg)", min_value=0.0, value=10.0)
        input_data['Tinggi'] = st.number_input("Tinggi Badan (cm)", min_value=0.0, value=75.0)
        input_data['LiLA'] = st.number_input("LiLA (cm)", min_value=0.0, value=12.5)

        # Konversi ke format DataFrame untuk prediksi
        input_df = pd.DataFrame([{
            'JK': 1 if input_data['JK'] == 'Perempuan' else 0,
            'Usia': input_data['Usia'],
            'Berat': input_data['Berat'],
            'Tinggi': input_data['Tinggi'],
            'LiLA': input_data['LiLA']
        }])

        # Tombol Prediksi
        if st.button("Prediksi"):
            model = st.session_state.adaboost
            pred_result = model.predict(input_df)[0]
            st.success(f"**Hasil Prediksi Status Gizi:** {pred_result}")
            