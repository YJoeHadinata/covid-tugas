## Analisis Cross-Sectional Antar Negara

### Gambaran Umum
Analisis cross-sectional mengambil snapshot data dari tanggal tertentu (2022-05-01) dan menganalisis semua negara secara bersamaan, bukan sebagai time series.

### 3D Scatter Plot Semua Negara
![Analisis Cross-Sectional 3D](cross_sectional_3d_analysis.png)
Plot 3D menampilkan hubungan antara:
- Total Cases per Million (X)
- Total Deaths per Million (Y)
- Total Vaccinations per Hundred (Z)

Indonesia ditandai dengan marker merah untuk perbandingan.

### Prediksi Antar Negara

#### Prediksi Kematian dari Kasus (Cross-Sectional):
- R² Score (test): 0.196
- Koefisien: 0.0045
- Interpretasi: Hubungan positif lemah antara kasus dan kematian antar negara

#### Prediksi Kematian dari Vaksinasi (Cross-Sectional):
- R² Score (test): 0.078
- Koefisien: 6.47
- Interpretasi: Vaksinasi berkorelasi positif dengan kematian (kemungkinan confounding)

#### Prediksi Multivariate (Kasus + Vaksinasi → Kematian):
- R² Score (test): 0.413
- Koefisien: [1.094, 0.0047]
- Interpretasi: Model terbaik untuk prediksi cross-sectional

## Kesimpulan
Analisis bivariate menunjukkan pola evolusi hubungan antara variabel COVID-19 dari waktu ke waktu. Model prediksi sederhana memberikan hasil yang bervariasi, dengan performa terbaik pada prediksi kasus dari kebijakan menggunakan Linear Regression.

## File Utama
- `covid_analysis.py`: Script analisis dan prediksi
- Berbagai file PNG untuk grafik hasil analisis

## Cara Menjalankan
```bash
python3 covid_analysis.py
```

Script akan menghasilkan grafik dan output prediksi di terminal.