# Analisis COVID-19: Bivariate dan Prediksi

Proyek ini melakukan analisis bivariate antara variabel COVID-19 untuk Indonesia dan United Kingdom, serta implementasi model prediksi sederhana menggunakan Linear Regression dan Random Forest.

## Data
- `covid.csv`: Dataset COVID-19 dari OWID (Our World in Data)

## Analisis Bivariate
Menganalisis hubungan antara variabel seperti:
- Kasus baru vs kematian baru
- Indeks ketat kebijakan vs kematian/kasus
- Vaksinasi vs reproduksi rate
- Vaksinasi vs kematian

Grafik scatter plot dengan pewarnaan berdasarkan waktu (colormap plasma) untuk menunjukkan evolusi hubungan dari waktu ke waktu.

## Prediksi
1. **Prediksi Kematian dari Kasus**: Menggunakan Linear Regression untuk memprediksi new_deaths dari new_cases
2. **Prediksi Kasus dari Kebijakan**: Menggunakan Linear Regression dan Random Forest untuk memprediksi new_cases dari stringency_index, new_vaccinations, dan new_tests

## File Utama
- `covid_analysis.py`: Script analisis dan prediksi
- Berbagai file PNG untuk grafik hasil analisis

## Cara Menjalankan
```bash
python3 covid_analysis.py
```

Script akan menghasilkan grafik dan output prediksi di terminal.