# Penerapan Hybrid CNN-LSTM untuk Prediksi Harga Emas

**Penulis:** [Nama Penulis]
**Afiliasi:** [Nama Institusi]
**Email:** [Alamat Email]

---

## Abstrak

Emas merupakan salah satu instrumen investasi yang paling diminati karena nilainya yang cenderung stabil dan tahan terhadap inflasi. Namun, fluktuasi harga emas yang dipengaruhi oleh berbagai faktor ekonomi global menuntut adanya metode prediksi yang akurat untuk membantu investor dalam pengambilan keputusan. Penelitian ini mengusulkan penerapan model *Deep Learning* hibrida, yaitu *Convolutional Neural Network* (CNN) dan *Long Short-Term Memory* (LSTM), untuk memprediksi harga emas harian. Model CNN digunakan untuk mengekstraksi fitur spasial dari data deret waktu, sedangkan LSTM bertugas menangkap ketergantungan temporal jangka panjang. Kinerja model Hybrid CNN-LSTM mengevaluasi menggunakan metrik *Mean Absolute Error* (MAE) dan *Root Mean Squared Error* (RMSE), serta dibandingkan dengan model *baseline* seperti Regresi Linear dan ARIMA. Hasil eksperimen menunjukkan bahwa model Hybrid CNN-LSTM mampu menangkap pola data dengan baik, menghasilkan MAE sebesar 0.0944 dan RMSE sebesar 0.1161, jauh lebih baik dibandingkan ARIMA (MAE 0.2052, RMSE 0.2493). Meskipun demikian, model Regresi Linear menunjukkan kinerja terbaik pada dataset ini (MAE 0.0131, RMSE 0.0194), mengindikasikan bahwa tren harga pada periode pengujian bersifat sangat linear. Penelitian ini memberikan wawasan mengenai potensi dan batasan model hibrida dalam prediksi harga komoditas.

**Kata Kunci:** Prediksi Harga Emas, CNN-LSTM, Deep Learning, Time Series Forecasting.

## Abstract

*Gold is one of the most sought-after investment instruments due to its stable value and resistance to inflation. However, gold price fluctuations influenced by various global economic factors require accurate prediction methods to assist investors in decision-making. This study proposes the application of a hybrid Deep Learning model, namely Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM), to predict daily gold prices. The CNN model is used to extract spatial features from time-series data, while LSTM captures long-term temporal dependencies. The performance of the Hybrid CNN-LSTM model is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics, and compared with baseline models such as Linear Regression and ARIMA. Experimental results show that the Hybrid CNN-LSTM model captures data patterns well, yielding an MAE of 0.0944 and an RMSE of 0.1161, significantly outperforming ARIMA (MAE 0.2052, RMSE 0.2493). Nevertheless, the Linear Regression model demonstrated the best performance on this dataset (MAE 0.0131, RMSE 0.0194), indicating that price trends in the testing period were highly linear. This study provides insights into the potential and limitations of hybrid models in commodity price prediction.*

**Keywords:** *Gold Price Prediction, CNN-LSTM, Deep Learning, Time Series Forecasting.*

---

## I. Pendahuluan

Investasi emas telah lama dianggap sebagai *safe haven* atau aset pelindung nilai di tengah ketidakpastian ekonomi global. Harga emas sangat dipengaruhi oleh berbagai faktor makroekonomi, seperti inflasi, suku bunga bank sentral, dan kondisi geopolitik [1]. Oleh karena itu, kemampuan untuk memprediksi pergerakan harga emas secara akurat menjadi sangat penting bagi investor individu, institusi keuangan, dan pembuat kebijakan.

Metode statistik tradisional seperti *AutoRegressive Integrated Moving Average* (ARIMA) dan Regresi Linear telah banyak digunakan dalam peramalan harga aset. Meskipun metode-metode ini efektif untuk data yang stasioner dan linear, mereka seringkali kesulitan dalam menangkap pola non-linear yang kompleks dan volatilitas tinggi yang sering terjadi pada pasar komoditas [2].

Dalam beberapa tahun terakhir, pendekatan *Deep Learning* telah menunjukkan kinerja yang superior dalam analisis deret waktu (*time series*). *Convolutional Neural Network* (CNN), yang awalnya dikembangkan untuk pengolahan citra, terbukti efektif dalam mengekstraksi fitur penting dari data [3]. Di sisi lain, *Long Short-Term Memory* (LSTM), varian dari *Recurrent Neural Network* (RNN), dirancang khusus untuk menangani masalah ketergantungan jangka panjang dan *vanishing gradient* pada data sekuensial [4].

Penelitian ini mengusulkan model hibrida yang menggabungkan keunggulan CNN dan LSTM (Hybrid CNN-LSTM) untuk prediksi harga emas. Dalam arsitektur ini, CNN berfungsi sebagai ekstraktor fitur yang menyaring informasi penting dari input deret waktu, yang kemudian diproses oleh lapisan LSTM untuk mempelajari pola temporalnya [5]. Tujuan dari penelitian ini adalah untuk mengevaluasi efektivitas model hibrida ini dan membandingkan kinerjanya dengan model tradisional (ARIMA dan Regresi Linear) pada dataset harga emas historis.

## II. Metodologi Penelitian

### 2.1 Dataset
Data yang digunakan dalam penelitian ini adalah data harian harga emas. Atribut yang digunakan untuk proses pelatihan dan prediksi adalah harga penutupan (*Close price*), karena dianggap mencerminkan nilai konsensus pasar pada akhir hari perdagangan.

### 2.2 Preprocessing Data
Sebelum dimasukkan ke dalam model, data melalui beberapa tahapan pra-pemrosesan:
1.  **Normalisasi**: Data harga dinormalisasi menggunakan teknik *Min-Max Scaling* untuk mengubah rentang nilai menjadi antara 0 dan 1. Hal ini penting untuk mempercepat konvergensi pelatihan model neural network [6].
2.  **Pembentukan Sekuens**: Data deret waktu diubah menjadi format *supervised learning* menggunakan metode *sliding window*.

### 2.3 Arsitektur Model Hybrid CNN-LSTM
Model yang dibangun terdiri dari beberapa lapisan utama sebagai berikut:
1.  **Input Layer**: Menerima sekuens data harga historis.
2.  **Convolutional Layer (Conv1D)**: Menggunakan 64 filter dengan ukuran kernel 3 dan fungsi aktivasi ReLU. Lapisan ini bertugas mengekstrak fitur lokal dari tren harga jangka pendek.
3.  **MaxPooling1D**: Dengan ukuran *pool* 2, digunakan untuk mereduksi dimensi fitur (*down-sampling*) dan mengurangi kompleksitas komputasi.
4.  **Dropout**: Sebesar 0.2 ditambahkan untuk mencegah *overfitting* selama proses pelatihan.
5.  **LSTM Layers**: Terdiri dari dua lapisan LSTM berturut-turut, masing-masing dengan 50 unit memori. Lapisan pertama mengembalikan sekuens (*return sequences=True*) untuk diteruskan ke lapisan LSTM kedua, yang bertugas menangkap pola ketergantungan jangka panjang.
6.  **Dense Layers**: Lapisan *fully connected* (Dense) dengan 25 neuron dan aktivasi ReLU, diikuti oleh lapisan output dengan 1 neuron untuk memprediksi harga emas pada satu langkah waktu ke depan.

Model dikompilasi menggunakan *optimizer* Adam dan fungsi kerugian (*loss function*) *Mean Squared Error* (MSE) [7].

## III. Hasil dan Pembahasan

### 3.1 Implementasi dan Pelatihan
Model dilatih selama 50 *epoch* dengan ukuran *batch* 32. Data dibagi menjadi data latih dan data uji dengan proporsi 90:10. Proses validasi dilakukan selama pelatihan untuk memantau kinerja model dan mencegah *overfitting*.

### 3.2 Evaluasi Model
Kinerja model dievaluasi menggunakan dua metrik utama: *Mean Absolute Error* (MAE) dan *Root Mean Squared Error* (RMSE). Tabel 1 menunjukkan perbandingan kinerja model Hybrid CNN-LSTM dengan model *baseline* (Regresi Linear dana ARIMA).

**Tabel 1. Perbandingan Kinerja Model**

| Model | MAE | RMSE |
| :--- | :--- | :--- |
| Linear Regression | 0.0131 | 0.0194 |
| ARIMA | 0.2052 | 0.2493 |
| **Hybrid CNN-LSTM** | **0.0944** | **0.1161** |

### 3.3 Diskusi
Berdasarkan hasil eksperimen, model Hybrid CNN-LSTM menunjukkan kinerja yang cukup baik dengan 0.0944 dan RMSE 0.1161. Model ini secara signifikan mengungguli model ARIMA (MAE 0.2052, RMSE 0.2493), membuktikan bahwa pendekatan *deep learning* mampu menangkap pola data yang lebih kompleks dibandingkan model statistik autoregresif standar pada dataset ini.

Namun, menarik untuk dicatat bahwa Regresi Linear memberikan hasil eror terendah (MAE 0.0131). Hal ini mengindikasikan bahwa data uji yang digunakan memiliki karakteristik tren linear yang sangat kuat, sehingga model sederhana cukup untuk melakukan prediksi akurat [8]. Meskipun demikian, keunggulan CNN-LSTM terletak pada kemampuannya untuk beradaptasi dengan pola non-linear. Dalam skenario pasar yang lebih fluktuatif (volatil), model hibrida diharapkan lebih tangguh (*robust*) dibandingkan regresi linear [9], [10].

## IV. Kesimpulan

Penelitian ini berhasil menerapkan model hibrida CNN-LSTM untuk prediksi harga emas. Penggabungan lapisan konvolusi untuk ekstraksi fitur dan LSTM untuk memori jangka panjang terbukti efektif, menghasilkan akurasi yang lebih baik daripada ARIMA. Meskipun Regresi Linear unggul pada dataset pengujian spesifik ini karena sifat linearitas data, arsitektur CNN-LSTM menawarkan potensi besar untuk menangani kompleksitas pasar yang dinamis. Penelitian selanjutnya dapat difokuskan pada penyetelan *hyperparameter* lebih lanjut dan penambahan variabel eksternal seperti nilai tukar mata uang dan indeks pasar saham untuk meningkatkan akurasi prediksi.

## Daftar Pustaka

[1] Livieris, I. E., Pintelas, E., & Pintelas, P. (2020). A CNN-LSTM model for gold price time-series forecasting. *Neural Computing and Applications*, 32(23), 17351-17360.

[2] Santika, R. R., & Hansun, S. (2021). Gold Price Prediction using CNN-LSTM. *2021 International Conference on Artificial Intelligence and Mechatronics Systems (AIMS)*, 1-5. It Bandung.

[3] Vidal, A., & Kristjanpoller, W. (2020). Gold price volatility forecasting: A hybrid approach. *Applied Economics*, 52(52), 5767-5778.

[4] Yamacli, S., & Canayaz, M. (2020). Architecture selection of CNN-LSTM for time series classification. *Measurement*, 153, 114674.

[5] Amini, M., & Kalantari, A. (2024). Gold Price Prediction Using A Hybrid CNN-Bi-LSTM Model. *Research Square*.

[6] Nastiti, P. (2021). Prediksi Harga Emas Menggunakan Metode Deep Learning dengan Implementasi TensorFlow. *Jurnal Teknologi Informasi dan Ilmu Komputer*, 8(2), 221-228.

[7] Bukhari, A. H., et al. (2020). Fractional LSTM: A new deep learning approach for time series forecasting. *Applied Sciences*, 10(21), 7436.

[8]  Yousuf, R., et al. (2022). Gold Price Prediction Using Ensemble Learning Techniques. *2022 IEEE 9th International Conference on Sciences of Electronics, Technologies of Information and Telecommunications (SETIT)*, 283-288.

[9]  Gabralla, L. A., & Abraham, A. (2020). Computational intelligence measurement of gold price volatility: A survey. *International Journal of Systems Assurance Engineering and Management*, 11, 1026-1035.

[10] Chen, J. (2021). A Hybrid CNN-LSTM Model for Forecasting Gold Price. *Journal of Physics: Conference Series*, 1802(3), 032049.
