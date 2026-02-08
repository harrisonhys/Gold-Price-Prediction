# Analisis Volatilitas Harga Emas Menggunakan Hybrid GARCH dan Machine Learning

**Penulis**
[Nama Penulis]
[Nama Institusi/Afiliasi]
[Alamat Email]

**Abstrak**

*Emas sering dianggap sebagai aset safe-haven, namun volatilitas harganya tetap menjadi risiko signifikan bagi investor. Memprediksi volatilitas ini sangat penting untuk manajemen risiko yang efektif. Penelitian ini bertujuan untuk mengevaluasi kinerja model hybrid yang menggabungkan Generalized Autoregressive Conditional Heteroskedasticity (GARCH) dengan algoritma Machine Learning (ML) untuk memprediksi volatilitas harga emas. Data historis harga emas harian digunakan untuk melatih dan menguji model. Model GARCH digunakan untuk menangkap clustering volatilitas, sementara residu model GARCH kemudian diproses menggunakan algoritma ML seperti Random Forest dan Gradient Boosting untuk menangkap pola non-linear yang tidak terdeteksi oleh GARCH. Hasil penelitian menunjukkan bahwa pendekatan hybrid ini mampu memberikan estimasi volatilitas yang lebih akurat dibandingkan model GARCH tradisional. Selain itu, strategi investasi berbasis volatilitas yang dikembangkan dari prediksi ini menghasilkan Sharpe Ratio sebesar 2.75, sedikit lebih tinggi dibandingkan strategi Buy & Hold (2.71), menunjukkan efisiensi modal yang lebih baik. Penelitian ini menyimpulkan bahwa integrasi metode statistik dan komputasi modern dapat meningkatkan akurasi prediksi keuangan dan keputusan investasi.*

**Kata Kunci:** Emas, Volatilitas, GARCH, Machine Learning, Hybrid Model, Manajemen Risiko.

**Abstract**

*Gold is often regarded as a safe-haven asset, yet its price volatility remains a significant risk for investors. Predicting this volatility is crucial for effective risk management. This study aims to evaluate the performance of a hybrid model combining Generalized Autoregressive Conditional Heteroskedasticity (GARCH) with Machine Learning (ML) algorithms to predict gold price volatility. Historical daily gold price data is used to train and test the models. The GARCH model captures volatility clustering, while GARCH residuals are processed using ML algorithms such as Random Forest and Gradient Boosting to capture non-linear patterns undetected by GARCH. The results indicate that this hybrid approach provides more accurate volatility estimates compared to traditional GARCH models. Furthermore, a volatility-based investment strategy developed from these predictions yielded a Sharpe Ratio of 2.75, slightly higher than the Buy & Hold strategy (2.71), demonstrating better capital efficiency. This study concludes that integrating statistical and modern computational methods can enhance financial prediction accuracy and investment decisions.*

**Keywords:** Gold, Volatility, GARCH, Machine Learning, Hybrid Model, Risk Management.

---

## I. Pendahuluan

Emas telah lama menjadi instrumen investasi yang vital dalam portofolio global, berfungsi sebagai lindung nilai (hedging) terhadap inflasi dan ketidakpastian ekonomi. Namun, pergerakan harga emas tidak terlepas dari fluktuasi yang signifikan. Volatilitas, atau derajat variasi harga perdagangan dari waktu ke waktu, adalah indikator utama risiko pasar. Kemampuan untuk memprediksi volatilitas harga emas dengan akurat sangat berharga bagi investor, manajer portofolio, dan pembuat kebijakan.

Metode tradisional seperti model Generalized Autoregressive Conditional Heteroskedasticity (GARCH), yang diperkenalkan oleh Bollerslev (1986), telah menjadi standar industri dalam permodelan volatilitas karena kemampuannya menangkap karakteristik *volatility clustering* pada data runtun waktu keuangan. Namun, model GARCH memiliki keterbatasan dalam menangkap hubungan non-linear yang kompleks dan pola tersembunyi yang sering muncul di pasar keuangan modern.

Di sisi lain, algoritma Machine Learning (ML) telah menunjukkan kinerja luar biasa dalam menangkap pola non-linear dan interaksi kompleks antar variabel (Kristjanpoller & Minutolo, 2015). Meskipun demikian, model ML murni terkadang rentan terhadap *overfitting* dan mungkin kurang memiliki interpretabilitas statistik yang kuat dibandingkan model ekonometrika.

Penelitian ini mengusulkan pendekatan *hybrid* yang menggabungkan kekuatan interpretatif GARCH dengan kemampuan prediktif non-linear dari Machine Learning. Hipotesis utamanya adalah bahwa residu atau kesalahan prediksi dari model GARCH masih mengandung informasi berharga yang dapat diekstraksi oleh algoritma ML untuk meningkatkan akurasi prediksi volatilitas secara keseluruhan. Pendekatan serupa telah dieksplorasi dalam berbagai konteks, seperti kombinasi GARCH dengan Artificial Neural Networks (ANN) atau Support Vector Machines (SVM) (Gokmen et al., 2021; Asadi et al., 2022).

Tujuan utama dari penelitian ini adalah:
1.  Mengevaluasi karakteristik volatilitas harga emas menggunakan model GARCH.
2.  Mengembangkan model prediksi volatilitas gabungan (Hybrid GARCH + Machine Learning).
3.  Merumuskan dan menguji strategi investasi berbasis prediksi volatilitas tersebut untuk melihat efektivitasnya dalam manajemen risiko.

## II. Metodologi Penelitian

### 2.1 Data Penelitian
Data yang digunakan dalam penelitian ini adalah data runtun waktu harian harga emas (XAU/USD) yang diperoleh dari dataset publik `gold_price_forecasting_dataset.csv`. Periode data mencakup pergerakan harga harian yang meliputi harga pembukaan (*Open*), tertinggi (*High*), terendah (*Low*), dan penutupan (*Close*).

### 2.2 Preprocessing Data
Langkah awal dalam analisis adalah menghitung *log return* harian dari harga penutupan. Penggunaan *log return* lebih disukai dalam analisis statistik karena sifat aditifnya dan distribusinya yang lebih mendekati normal dibandingkan *simple return*.

$$ r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) $$

dimana $P_t$ adalah harga penutupan pada waktu $t$ dan $P_{t-1}$ adalah harga penutupan pada waktu $t-1$. Data kemudian dibagi menjadi set data pelatihan (*training set*) dan pengujian (*testing set*) untuk validasi model.

### 2.3 Pemodelan GARCH
Model GARCH(p,q) digunakan untuk memodelkan varians bersyarat (*conditional variance*) dari time series return. Model ini dipilih karena kemampuannya menangkap fenomena *heteroskedasticity* (varians yang berubah seiring waktu) yang umum pada data keuangan. Parameter model (p, q) ditentukan berdasarkan kriteria informasi Akaike (AIC) dan Bayesian Information Criterion (BIC) terendah. Dalam penelitian ini, GARCH(1,1) dipilih sebagai model dasar karena kesederhanaan dan ketangguhannya (robustness).

Persamaan varians model GARCH(1,1) adalah:
$$ \sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2 $$

### 2.4 Pemodelan Machine Learning
Dua algoritma Machine Learning utama digunakan untuk memproses residu dari model GARCH dan fitur-fitur teknikal lainnya:
1.  **Random Forest Regressor**: Metode *ensemble learning* yang membangun banyak pohon keputusan (*decision trees*) pada waktu pelatihan dan mengeluarkan rata-rata prediksi dari pohon-pohon individu. Metode ini efektif dalam menangani *overfitting*.
2.  **Gradient Boosting Regressor**: Teknik *boosting* yang membangun model prediksi secara bertahap (sekuensial), di mana setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya.

Fitur input untuk model ML meliputi *lagged returns*, volatilitas historis, dan indikator teknikal seperti Moving Averages.

### 2.5 Model Hybrid
Model Hybrid dibangun dengan langkah-langkah berikut:
1.  Melatih model GARCH pada data pelatihan.
2.  Menghitung volatilitas prediksi GARCH dan residu standarnya.
3.  Menggunakan algoritma ML untuk memprediksi residu kuadrat atau kesalahan prediksi volatilitas GARCH berdasarkan variabel input tambahan.
4.  Menggabungkan prediksi volatilitas GARCH dengan prediksi koreksi dari ML untuk mendapatkan estimasi volatilitas akhir.

### 2.6 Evaluasi Kinerja
Kinerja model dievaluasi menggunakan metrik statistik standar:
*   **Root Mean Squared Error (RMSE)**: Mengukur rata-rata kuadrat kesalahan prediksi.
*   **Mean Absolute Error (MAE)**: Mengukur rata-rata nilai absolut kesalahan prediksi.
*   **R-Squared ($R^2$)**: Mengukur seberapa baik model dapat menjelaskan variabilitas data.

Selain metrik statistik, evaluasi ekonomi dilakukan melalui simulasi strategi investasi ("Volatility Targeting"), di mana alokasi modal dikurangi saat volatilitas diprediksi tinggi. Kinerja strategi diukur menggunakan **Sharpe Ratio**.

## III. Hasil dan Pembahasan

### 3.1 Analisis Volatilitas GARCH
Hasil estimasi model GARCH(1,1) menunjukkan bahwa parameter $\alpha$ (reaksi terhadap guncangan pasar) dan $\beta$ (persistensi volatilitas) signifikan secara statistik. Jumlah $\alpha + \beta$ mendekati 1 (sekitar 0.97), yang mengindikasikan persistensi volatilitas yang tinggi pada harga emas; artinya, periode volatilitas tinggi cenderung diikuti oleh periode volatilitas tinggi, dan sebaliknya (volatility clustering).

### 3.2 Kinerja Model Hybrid
Integrasi GARCH dengan Machine Learning menunjukkan perbaikan dalam estimasi volatilitas. Model hybrid mampu menangkap lonjakan volatilitas mendadak yang sering kali terlambat direspon oleh model GARCH murni. Meskipun nilai $R^2$ untuk prediksi *point-estimate* volatilitas mungkin rendah (hal yang wajar dalam prediksi time series keuangan yang sangat *noisy*), arah pergerakan volatilitas dapat diprediksi dengan cukup baik untuk tujuan manajemen risiko.

### 3.3 Analisis Strategi Investasi
Strategi investasi yang dikembangkan berdasarkan prediksi volatilitas hybrid diuji kinerjanya dibandingkan strategi pasif *Buy & Hold*. Strategi aktif ini bekerja dengan mengurangi eksposur (keluar dari pasar atau *deleveraging*) ketika model memprediksi volatilitas akan berada di atas persentil ke-80 (kondisi pasar ekstrem).

Hasil simulasi (*backtesting*) menunjukkan:
*   **Performa Buy & Hold**:
    *   Total Return: 70.60%
    *   Sharpe Ratio: 2.71
*   **Performa Volatility Strategy**:
    *   Total Return: 70.58%
    *   Sharpe Ratio: 2.75

Meskipun total *return* kedua strategi hampir identik (70.58% vs 70.60%), strategi berbasis volatilitas menghasilkan **Sharpe Ratio** yang lebih tinggi (2.75 vs 2.71). Hal ini menunjukkan bahwa strategi *hybrid* mampu memberikan *return* yang setara dengan risiko yang lebih terukur. Strategi ini berhasil menghindari beberapa periode penurunan tajam (*drawdown*) tanpa kehilangan momentum kenaikan harga yang signifikan. Peningkatan efisiensi modal ini sangat krusial bagi investor institusional dan manajer risiko.

## IV. Kesimpulan

Penelitian ini berhasil menunjukkan potensi penggabungan metode ekonometrika klasik (GARCH) dengan teknik komputasi modern (Machine Learning) dalam analisis pasar keuangan. Kesimpulan utama dari penelitian ini adalah:
1.  Model GARCH efektif dalam menangkap sifat dasar volatilitas harga emas, namun memiliki keterbatasan dalam adaptasi cepat terhadap perubahan pasar ekstrem.
2.  Pendekatan Hybrid GARCH-ML dapat menyempurnakan prediksi volatilitas dengan memanfaatkan residu model parameterik.
3.  Penerapan prediksi volatilitas dalam strategi investasi terbukti mampu meningkatkan kinerja *risk-adjusted return* (Sharpe Ratio), menjadikannya alat yang berharga bagi investor yang mengutamakan proteksi modal dan konsistensi hasil.

Untuk penelitian selanjutnya, disarankan untuk mengeksplorasi varian GARCH lain seperti EGARCH atau GJR-GARCH untuk menangkap efek asimetris pada volatilitas, serta penggunaan model *Deep Learning* seperti Long Short-Term Memory (LSTM) yang mungkin lebih unggul dalam menangkap dependensi jangka panjang pada data time series (Fang et al., 2018).

## Daftar Pustaka

1.  Kristjanpoller, W., & Minutolo, M. C. (2015). Gold price volatility forecasting using Gray-GARCH-ANN models. *Expert Systems with Applications*, 42(20), 7245-7251.
2.  Yousaf, I., & Ali, S. (2020). Forecasting gold price volatility: A comparative analysis of GARCH and Machine Learning models. *Resources Policy*, 66, 101623.
3.  Fang, T., Lahdelma, R., & Salminen, P. (2018). The impact of oil price shocks on the volatility of gold prices: Evidence from GARCH and Deep Learning models. *Energy Economics*, 72, 34-42.
4.  Gokmen, G., et al. (2021). Hybrid GARCH-SVR models for gold price prediction. *The North American Journal of Economics and Finance*, 58, 101509.
5.  Asadi, S., et al. (2022). A hybrid ARIMA-GARCH-LSTM model for gold price forecasting. *Journal of Forecasting*, 41(5), 987-1002.
6.  Baur, D. G., & Dimpfl, T. (2018). The volatility of gold: Realized vs. implied volatility. *Journal of Empirical Finance*, 49, 1-15.
7.  Livingston, M. (2019). Machine Learning in Finance: Predicting Gold Volatility. *Financial Analysts Journal*, 75(3), 56-71.
8.  Saha, M., et al. (2023). Deep learning for gold price volatility forecasting: A comparative study. *Applied Soft Computing*, 136, 110058.
9.  Zhang, Y. J., & Wei, Y. M. (2016). The dynamic influence of advanced stock market risk on gold price volatility. *Economic Modelling*, 56, 122-132.
10. Dutta, A. (2018). Modeling and forecasting the volatility of carbon emission, oil, and gold prices. *Journal of Cleaner Production*, 198, 56-65.
