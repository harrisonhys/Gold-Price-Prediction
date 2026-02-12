# Prediksi Harga Emas Menggunakan Deep Learning dan Optimasi Metaheuristik

[English Version](#english-version)

## Deskripsi

Penelitian ini mengeksplorasi berbagai pendekatan komputasional dalam memprediksi pergerakan harga emas, sebuah komoditas strategis yang memiliki peran penting dalam sistem ekonomi global. Melalui implementasi beberapa arsitektur model pembelajaran mesin, proyek ini bertujuan untuk memberikan wawasan mengenai efektivitas berbagai teknik prediksi dalam konteks data time series finansial.

Harga emas dikenal sebagai instrumen yang kompleks dan dipengaruhi oleh berbagai faktor ekonomi makro, volatilitas pasar, dan sentimen investor. Oleh karena itu, pemodelan yang akurat memerlukan pendekatan yang mampu menangkap pola temporal, volatilitas, dan non-linearitas yang inheren dalam data tersebut.

## Metodologi

Proyek ini mengimplementasikan tiga pendekatan utama:

### 1. CNN-LSTM untuk Ekstraksi Fitur Spasial-Temporal
Pendekatan ini mengkombinasikan kekuatan Convolutional Neural Network (CNN) dalam mengekstraksi fitur spasial dari data time series dengan Long Short-Term Memory (LSTM) yang unggul dalam menangkap dependensi temporal jangka panjang. Arsitektur hybrid ini memungkinkan model untuk belajar representasi hierarkis dari pola-pola kompleks dalam pergerakan harga.

**Lokasi:** `CNN LSTM/`

**Komponen utama:**
- Notebook implementasi lengkap (`gold_price_prediction.ipynb`)
- Spesifikasi dependensi (`requirements.txt`)

### 2. GARCH-ML untuk Pemodelan Volatilitas
Generalized Autoregressive Conditional Heteroskedasticity (GARCH) merupakan pendekatan statistik yang telah terbukti efektif dalam memodelkan volatilitas time series finansial. Dalam proyek ini, GARCH dikombinasikan dengan algoritma pembelajaran mesin untuk tidak hanya memodelkan volatilitas tetapi juga mengembangkan strategi investasi yang diinformasikan oleh prediksi volatilitas tersebut.

**Lokasi:** `GARCH/`

**Komponen utama:**
- Analisis komprehensif volatilitas (`Gold_Price_Volatility_Analysis_GARCH_ML.ipynb`)
- Dokumentasi penelitian (`Journal_Article_GARCH_ML.md`)

### 3. Grey Wolf Optimizer-LSTM untuk Hyperparameter Tuning
Pendekatan ini mengintegrasikan algoritma optimasi metaheuristik Grey Wolf Optimizer (GWO) dengan LSTM. GWO, yang terinspirasi dari hierarki kepemimpinan dan mekanisme berburu serigala abu-abu, digunakan untuk mengoptimalkan hyperparameter LSTM secara otomatis, mengurangi ketergantungan pada tuning manual yang seringkali memakan waktu.

**Lokasi:** `Grey Wolf Optimizer/`

**Komponen utama:**
- Implementasi GWO-LSTM (`gold_price_gwo_lstm.ipynb`)
- Draft artikel penelitian (`Journal_Draft.md`)

## Dataset

Dataset yang digunakan dalam penelitian ini (`gold_price_forecasting_dataset.csv`) merupakan kumpulan data historis harga emas yang mencakup berbagai indikator ekonomi dan teknikal. Data ini telah dikurasi untuk memfasilitasi analisis time series dan pemodelan prediktif.

**Sumber Data:** [Gold Price Forecasting Dataset](https://www.kaggle.com/datasets/vishardmehta/gold-price-forecasting-dataset/data) oleh Vishard Mehta, tersedia di Kaggle.

Dataset ini menyediakan informasi komprehensif yang memungkinkan analisis terhadap faktor-faktor yang mempengaruhi pergerakan harga emas. Kami mengucapkan terima kasih kepada penyedia dataset atas kontribusinya dalam mendukung riset berbasis data terbuka.

## Struktur Direktori

```
.
├── CNN LSTM/
│   ├── gold_price_prediction.ipynb
│   ├── Journal_Article_CNN_LSTM.md
│   ├── main-idea.txt
│   └── requirements.txt
├── GARCH/
│   ├── Gold_Price_Volatility_Analysis_GARCH_ML.ipynb
│   └── Journal_Article_GARCH_ML.md
├── Grey Wolf Optimizer/
│   ├── gold_price_gwo_lstm.ipynb
│   └── Journal_Draft.md
├── template/
│   └── Template_Jurnal_JIT_2025.pdf
├── gold_price_forecasting_dataset.csv
└── read_pdf.py
```

## Persyaratan Sistem

### Lingkungan Python
- Python 3.8 atau lebih tinggi
- Jupyter Notebook atau JupyterLab

### Dependensi Utama
Setiap sub-proyek memiliki file `requirements.txt` masing-masing. Secara umum, dependensi meliputi:
- TensorFlow/Keras untuk implementasi deep learning
- Scikit-learn untuk preprocessing dan evaluasi
- Pandas dan NumPy untuk manipulasi data
- Matplotlib dan Seaborn untuk visualisasi
- ARCH untuk pemodelan GARCH

## Instalasi dan Penggunaan

### 1. Clone Repository
```bash
git clone https://github.com/harrisonhys/Gold-Price-Datasets.git
cd Gold-Price-Datasets
```

### 2. Buat Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Untuk Linux/Mac
# atau
.venv\Scripts\activate  # Untuk Windows
```

### 3. Instalasi Dependensi
Untuk CNN-LSTM:
```bash
cd "CNN LSTM"
pip install -r requirements.txt
```

Untuk pendekatan lain, sesuaikan direktori dan instalasi dependensi yang diperlukan.

### 4. Jalankan Notebook
```bash
jupyter notebook
```

Navigasi ke notebook yang diinginkan dan jalankan sel-sel secara berurutan.

## Hasil dan Temuan

Setiap pendekatan menghasilkan insight unik mengenai prediksi harga emas:

- **CNN-LSTM** menunjukkan kemampuan dalam menangkap pola kompleks melalui ekstraksi fitur hierarkis
- **GARCH-ML** memberikan pemahaman tentang dinamika volatilitas dan aplikasinya dalam strategi investasi
- **GWO-LSTM** mendemonstrasikan efektivitas optimasi metaheuristik dalam meningkatkan performa model

Dokumentasi lengkap mengenai hasil eksperimen, evaluasi model, dan analisis komparatif tersedia dalam artikel jurnal di masing-masing direktori.

## Kontribusi

Kontribusi untuk pengembangan lebih lanjut sangat diterima. Beberapa area yang dapat dieksplorasi:
- Implementasi arsitektur deep learning terkini (Transformer, Attention Mechanism)
- Integrasi data alternatif (sentimen media sosial, indikator makroekonomi tambahan)
- Pengembangan ensemble model untuk meningkatkan robustness prediksi
- Eksperimen dengan algoritma optimasi metaheuristik lainnya (PSO, GA, etc.)

## Lisensi

Proyek ini dikembangkan untuk tujuan penelitian dan edukasi. Silakan merujuk pada lisensi yang sesuai sebelum menggunakan untuk keperluan komersial.

## Kontak

Untuk pertanyaan, diskusi, atau kolaborasi penelitian, silakan hubungi melalui repository issues atau pull requests.

---

# English Version

## Gold Price Prediction Using Deep Learning and Metaheuristic Optimization

## Description

This research explores various computational approaches in predicting gold price movements, a strategic commodity that plays a crucial role in the global economic system. Through the implementation of several machine learning and deep learning architectures, this project aims to provide insights into the effectiveness of various prediction techniques in the context of financial time series data.

Gold prices are known to be complex instruments influenced by various macroeconomic factors, market volatility, and investor sentiment. Therefore, accurate modeling requires approaches capable of capturing temporal patterns, volatility, and non-linearity inherent in such data.

## Methodology

This project implements three main approaches:

### 1. CNN-LSTM for Spatio-Temporal Feature Extraction
This approach combines the strength of Convolutional Neural Networks (CNN) in extracting spatial features from time series data with Long Short-Term Memory (LSTM), which excels at capturing long-term temporal dependencies. This hybrid architecture enables the model to learn hierarchical representations of complex patterns in price movements.

**Location:** `CNN LSTM/`

**Main components:**
- Complete implementation notebook (`gold_price_prediction.ipynb`)
- In-depth analysis journal article (`Journal_Article_CNN_LSTM.md`)
- Dependency specifications (`requirements.txt`)

### 2. GARCH-ML for Volatility Modeling
Generalized Autoregressive Conditional Heteroskedasticity (GARCH) is a statistical approach proven effective in modeling financial time series volatility. In this project, GARCH is combined with machine learning algorithms not only to model volatility but also to develop investment strategies informed by these volatility predictions.

**Location:** `GARCH/`

**Main components:**
- Comprehensive volatility analysis (`Gold_Price_Volatility_Analysis_GARCH_ML.ipynb`)
- Research documentation (`Journal_Article_GARCH_ML.md`)

### 3. Grey Wolf Optimizer-LSTM for Hyperparameter Tuning
This approach integrates the metaheuristic optimization algorithm Grey Wolf Optimizer (GWO) with LSTM. GWO, inspired by the leadership hierarchy and hunting mechanism of grey wolves, is used to automatically optimize LSTM hyperparameters, reducing reliance on manual tuning which is often time-consuming.

**Location:** `Grey Wolf Optimizer/`

**Main components:**
- GWO-LSTM implementation (`gold_price_gwo_lstm.ipynb`)
- Research article draft (`Journal_Draft.md`)

## Dataset

The dataset used in this research (`gold_price_forecasting_dataset.csv`) is a collection of historical gold price data encompassing various economic and technical indicators. This data has been curated to facilitate time series analysis and predictive modeling.

**Data Source:** [Gold Price Forecasting Dataset](https://www.kaggle.com/datasets/vishardmehta/gold-price-forecasting-dataset/data) by Vishard Mehta, available on Kaggle.

This dataset provides comprehensive information enabling in-depth analysis of factors influencing gold price movements. We extend our gratitude to the dataset provider for their contribution in supporting open data-driven research.

## Directory Structure

```
.
├── CNN LSTM/
│   ├── gold_price_prediction.ipynb
│   ├── Journal_Article_CNN_LSTM.md
│   ├── main-idea.txt
│   └── requirements.txt
├── GARCH/
│   ├── Gold_Price_Volatility_Analysis_GARCH_ML.ipynb
│   └── Journal_Article_GARCH_ML.md
├── Grey Wolf Optimizer/
│   ├── gold_price_gwo_lstm.ipynb
│   └── Journal_Draft.md
├── template/
│   └── Template_Jurnal_JIT_2025.pdf
├── gold_price_forecasting_dataset.csv
└── read_pdf.py
```

## System Requirements

### Python Environment
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Main Dependencies
Each sub-project has its own `requirements.txt` file. Generally, dependencies include:
- TensorFlow/Keras for deep learning implementation
- Scikit-learn for preprocessing and evaluation
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for visualization
- ARCH for GARCH modeling

## Installation and Usage

### 1. Clone Repository
```bash
git clone https://github.com/harrisonhys/Gold-Price-Datasets.git
cd Gold-Price-Datasets
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # For Linux/Mac
# or
.venv\Scripts\activate  # For Windows
```

### 3. Install Dependencies
For CNN-LSTM:
```bash
cd "CNN LSTM"
pip install -r requirements.txt
```

For other approaches, adjust the directory and install the necessary dependencies accordingly.

### 4. Run Notebook
```bash
jupyter notebook
```

Navigate to the desired notebook and run the cells sequentially.

## Results and Findings

Each approach yields unique insights into gold price prediction:

- **CNN-LSTM** demonstrates capability in capturing complex patterns through hierarchical feature extraction
- **GARCH-ML** provides deep understanding of volatility dynamics and its application in investment strategies
- **GWO-LSTM** showcases the effectiveness of metaheuristic optimization in enhancing model performance

Complete documentation regarding experimental results, model evaluation, and comparative analysis is available in the journal articles within each directory.

## Contribution

Contributions for further development are warmly welcome. Some areas that can be explored:
- Implementation of state-of-the-art deep learning architectures (Transformer, Attention Mechanism)
- Integration of alternative data sources (social media sentiment, additional macroeconomic indicators)
- Development of ensemble models to improve prediction robustness
- Experimentation with other metaheuristic optimization algorithms (PSO, GA, etc.)

## License

This project is developed for research and educational purposes. Please refer to the appropriate license before using it for commercial purposes.

## Contact

For questions, discussions, or research collaboration, please reach out through repository issues or pull requests.
