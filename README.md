
<!---
sudeozen17/sudeozen17 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
data = pd.read_csv("/kaggle/input/apple-stock-data/apple_stock_data3", index_col="Date", parse_dates=True)
​
# Gerekli kütüphaneler
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
​
# Adım 1: Veri Yükleme
data = pd.read_csv("/kaggle/input/apple-stock-data/apple_stock_data.csv", index_col="Date", parse_dates=True)
​
# Kapanış fiyatlarını alalım
close_prices = data["Close"].values.reshape(-1, 1)
​
# Fiyatları 0-1 aralığına ölçekleyelim
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)
​
# Eğitim ve test verilerini ayırma (80% train, 20% test)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
​
# Zaman serisi veri seti oluşturma
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
​
# 60 zaman adımlı veri seti oluşturma
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)
​
# LSTM giriş verilerinin boyutunu ayarlama (örnek sayısı, zaman adımı, özellik sayısı)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
​
# Adım 2: LSTM Modeli Oluşturma
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
​
model.compile(optimizer="adam", loss="mean_squared_error")
​
# Modeli eğitme
model.fit(X_train, y_train, epochs=5, batch_size=32)
​
# Adım 3: Tahmin Yapma
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
​
# Fiyatları orijinal aralığına geri döndürme
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
​
# Adım 4: Sonuçları Görselleştirme
plt.figure(figsize=(14,5))
plt.plot(data.index[60:train_size], train_predict, label="Eğitim Tahminleri", color='blue')
plt.plot(data.index[train_size+60:], test_predict, label="Test Tahminleri", color='red')
plt.plot(data.index, data['Close'], label="Gerçek Fiyat", color='green')
plt.title("Apple Hisse Fiyat Tahmini")
plt.xlabel("Tarih")
plt.ylabel("Fiyat (USD)")
plt.legend()
plt.show()
​
add Codeadd Markdown
arrow_upwardarrow_downwarddelete
Apple Hisse Senedi Zaman Serisi Tahmini - LSTM Modeli

Bu proje, Apple Inc.'in hisse senedi kapanış fiyatlarını tahmin etmek için bir LSTM (Long Short-Term Memory) modeli kullanır. LSTM, zaman serisi verilerindeki uzun dönemli bağımlılıkları yakalayabilen bir tür tekrarlayan sinir ağı (RNN) modelidir. Proje, Kaggle Notebook ortamında çalıştırılmak üzere hazırlanmıştır.

İçindekiler

Proje Hakkında
Gereksinimler
Veri Seti
Model Mimari ve Yöntemler
Sonuçlar
Kullanım
Proje Hakkında

Bu projede, Apple'ın 2020-2024 yılları arasındaki hisse senedi kapanış fiyatları kullanılarak, gelecekteki fiyatların tahmin edilmesi amaçlanmıştır. Model, geçmiş 60 günün fiyatlarını kullanarak sonraki günün fiyatını tahmin eder.

Kullanılan Teknikler:

Zaman Serisi Tahmini: Kapanış fiyatlarını zaman serisi analizine tabi tutarak tahmin yapma.
LSTM (Long Short-Term Memory): Zaman serisi verileri üzerinde başarılı olan tekrarlayan sinir ağı türü.
Keras ve TensorFlow: Model oluşturma ve eğitme için.
Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphaneler gereklidir:

numpy
pandas
scikit-learn
matplotlib
tensorflow
keras
Bu kütüphaneler Kaggle Notebook ortamında önceden yüklü olarak gelir.

Veri Seti

Veri seti, Apple Inc.'in hisse senedi kapanış fiyatlarını içerir ve Kaggle'da bir CSV dosyası olarak yüklenmiştir. Bu veri seti şunları içerir:

Tarih (Date): Hisse senedi verisinin alındığı tarih.
Kapanış Fiyatı (Close): Apple'ın günlük kapanış fiyatı (USD).
Veriyi indirebilirsiniz: Apple Stock Data - CSV

Model Mimari ve Yöntemler

1. Veri Ön İşleme:

Ölçekleme: Kapanış fiyatları, LSTM modeline uygun hale getirmek için 0-1 aralığına ölçeklenmiştir.
Veri Ayırma: Verinin %80’i eğitim, %20’si test seti olarak ayrılmıştır.
Zaman Adımları: Modelin eğitimi için geçmiş 60 günlük veriyi kullanarak sonraki günün fiyatı tahmin edilmektedir.
2. Model Mimari:

İki katmanlı LSTM yapısı:
İlk LSTM katmanı, 50 birim ve return_sequences=True parametresi ile, sıralı veriyi sonraki LSTM katmanına aktarır.
İkinci LSTM katmanı yine 50 birim ile son tahmini sağlar.
Çıkış katmanı: Tek bir çıkış birimi kullanılarak sonraki kapanış fiyatı tahmin edilir.
3. Modelin Eğitimi:

Loss Function: Mean Squared Error (MSE) kullanılmıştır.
Optimizer: Adam optimizasyon algoritması kullanılmıştır.
Epochs: Model 5 epoch boyunca eğitilmiştir.
Batch Size: 32 örnekten oluşan mini-batch ile eğitim yapılmıştır.
Sonuçlar

Model eğitim sürecinin ardından hem eğitim hem de test veri setleri üzerindeki tahmin sonuçları, görselleştirilmiş ve gerçek fiyatlarla karşılaştırılmıştır. Sonuçlar aşağıdaki gibi görselleştirilir:

Mavi: Eğitim setindeki tahminler.
Kırmızı: Test setindeki tahminler.
Yeşil: Gerçek fiyatlar.
Kullanım

1. Kaggle Ortamında Projenin Çalıştırılması:

Adım 1: Kaggle'a giriş yapın ve yeni bir Notebook oluşturun.
Adım 2: Veri setini Kaggle'a yükleyin veya veri setini doğrudan Kaggle'da bir dataset olarak kullanın.
Adım 3: Yukarıda belirtilen LSTM model kodunu Notebook'a yapıştırın.
Adım 4: Tüm hücreleri çalıştırarak modelin eğitim ve tahmin sonuçlarını gözlemleyin.
2. Local Ortamda Kullanım:

Local ortamda kullanmak için, gerekli Python kütüphanelerini yükleyin.
Kaggle notebook'u veya .ipynb dosyasını indirip Jupyter Notebook üzerinden çalıştırın.
