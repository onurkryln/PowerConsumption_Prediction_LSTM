import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
# 1. VERİ OKUMA VE ÖN İŞLEME
# ----------------------------------------------------------------
veri = pd.read_csv("powerconsumption.csv")

# HATA DÜZELTME 1: Satır sonundaki görünmez karakter temizlendi
veri["Datetime"] = pd.to_datetime(veri["Datetime"])

veri = veri.sort_values("Datetime")
veri.set_index("Datetime", inplace=True)

# Frekans ayarlama (10 dakikalık)
veri = veri.asfreq("10min")
veri = veri.interpolate(method='linear')

# Toplam tüketim hesaplama
veri['TotalPowerConsumption'] = (veri['PowerConsumption_Zone1'] + 
                                 veri['PowerConsumption_Zone2'] + 
                                 veri['PowerConsumption_Zone3'])

# Gereksiz Zone sütunlarını at, AMA HAVA DURUMU sütunlarını koru!
veri = veri.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1)

# 2. YENİDEN ÖRNEKLEME (RESAMPLING)
# ----------------------------------------------------------------
# Saatlik Veri (Analiz ve Boxplot için bunu kullanacağız)
#saat_veri = veri.resample('h').mean()



# 3. GÖRSELLEŞTİRME 1: DECOMPOSITION (MEVSİMSEL AYRIŞTIRMA)
# ----------------------------------------------------------------
# Decomposition genellikle Günlük veri üzerinden daha temiz görünür
decomposition = seasonal_decompose(veri["TotalPowerConsumption"], model='additive', period=144)
plt.rcParams['figure.figsize'] = (10, 8)
decomposition.plot()
plt.show()

# 4. GÖRSELLEŞTİRME 2: KUTU GRAFİKLERİ (BOXPLOTS)
# ----------------------------------------------------------------
# HATA DÜZELTME 2: Boxplot için 'günlük_veri' değil 'saat_veri' kullanıyoruz.
# Çünkü günlük veride saat hep 00:00'dır, saatlik değişim görülmez.

temp_df =veri.copy() # <-- BURASI DEĞİŞTİ (günlük_veri -> saat_veri)
veri['month'] = temp_df.index.month
# Saati döngüsel (cyclic) hale getirme
veri['hour_sin'] = np.sin(2 * np.pi * veri.index.hour / 24)
veri['hour_cos'] = np.cos(2 * np.pi * veri.index.hour / 24)

# Bu yeni sütunları da features listesine manuel eklemek isteyebilirsiniz.
# temp_df['day_of_week'] = temp_df.index.dayofweek # İsterseniz ekleyebilirsiniz

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Saatlik Tüketim (Artık saatler 0-23 arası görünecek)
sns.boxplot(x='hour_sin', y='TotalPowerConsumption', data=veri, ax=axes[0])
axes[0].set_title('Saatlik Tüketim Dağılımı')

# Aylık Tüketim
sns.boxplot(x='month', y='TotalPowerConsumption', data=veri, ax=axes[1])
axes[1].set_title('Aylık Tüketim Dağılımı')

plt.tight_layout()
plt.show()

# 5. GÖRSELLEŞTİRME 3: OTOKORELASYON (ACF / PACF)
# ----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(veri['TotalPowerConsumption'].dropna(), lags=30, ax=ax[0], title="ACF (Günlük)")
plot_pacf(veri['TotalPowerConsumption'].dropna(), lags=30, ax=ax[1], title="PACF (Günlük)")
plt.show()

# 6. KORELASYON ANALİZİ
# ----------------------------------------------------------------
# Korelasyonu Saatlik veri (temp_df) üzerinden yapıyoruz ki 'hour' etkisi görülsün.
plt.figure(figsize=(10, 8))
sns.heatmap(veri.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Isı Haritası (Saatlik Veri)")
plt.show()

target_col = 'TotalPowerConsumption'
cols = [c for c in veri.columns if c != target_col] + [target_col]
veri = veri[cols]

# Nitelik Seçimi (0.1 eşik değeri ile)
#corr = veri.corr()["TotalPowerConsumption"].sort_values(ascending=False)
#ek_nitelikler = corr[(abs(corr) > 0.2) & (corr.index != "TotalPowerConsumption")].index.tolist()

lags_to_check = [1, 3, 6, 12, 24, 72, 144] 
THRESHOLD = 0.3 # Korelasyon eşik değeri (Bunun altındakileri ele)

selected_raw_features = [] # Seçilen ham sütunlar burada toplanacak
candidate_features=veri
candidate_features=candidate_features.drop(["TotalPowerConsumption"],axis=1)



for col in candidate_features:
    max_corr = 0
    best_lag = 0
    
    # Her bir lag için korelasyona bak
    for lag in lags_to_check:
        # Değişkeni kaydırıp hedefle ilişkisine bakıyoruz
        shifted_col = veri[col].shift(lag)
        corr = veri["TotalPowerConsumption"].corr(shifted_col)
        
        # En yüksek korelasyonu (mutlak değer olarak) sakla
        if abs(corr) > abs(max_corr):
            max_corr = corr
            best_lag = lag
            
    print(f"Değişken: {col:12} | Maks Korelasyon: {max_corr:.4f} (Lag: {best_lag})")
    
    # Eşik değerini geçiyorsa seçilenlere ekle
    if abs(max_corr) > THRESHOLD:
        selected_raw_features.append(col)

# Target'ı da listeye ekle (O her zaman lazım)
final_features = selected_raw_features + ["TotalPowerConsumption"]

print("\n--- KORELASYON ANALİZİ SONUCU ---")
print("Tüm Korelasyonlar:\n", final_features)
print("-" * 30)
#print("Seçilen Nitelikler (>0.1):\n", ek_nitelikler)

veri=veri[final_features]
trainsize = int(len(veri) * 0.8)
train_data = veri.iloc[:trainsize] # train_data değişkeni burada tanımlandı!
test_data = veri.iloc[trainsize:]

print(f"Eğitim Verisi Boyutu: {train_data.shape}")
print(f"Test Verisi Boyutu: {test_data.shape}")

# 5. ÖLÇEKLEME (SCALING)
# ----------------------------------------------------------------
sc = MinMaxScaler(feature_range=(0, 1))

# Sadece Train üzerine fit edilir
train_scaled = sc.fit_transform(train_data)
test_scaled = sc.transform(test_data)

# 6. VERİYİ LSTM FORMATINA (LAG) ÇEVİRME
# ----------------------------------------------------------------
def create_lag(dataset, lag=6):
    X, y = [], []
    for i in range(lag, len(dataset)):
        X.append(dataset[i-lag:i, :]) 
        y.append(dataset[i, -1]) # Hedef değişken (TotalPowerConsumption) 0. sütunda varsayıyoruz
    return np.array(X), np.array(y)

lag_step =24
x_train, y_train = create_lag(train_scaled, lag_step)
x_test, y_test = create_lag(test_scaled, lag_step)

print(f"LSTM Girdi Şekli (x_train): {x_train.shape}")
# Örn: (Örnek Sayısı, 30, Özellik Sayısı)

# 7. MODEL KURULUMU
# ----------------------------------------------------------------
model = Sequential()

# 1. LSTM Katmanı
# input_shape=(Zaman Adımı, Özellik Sayısı) -> (30, feature_count)
model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))

# 2. LSTM Katmanı
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))

# Çıkış Katmanı
model.add(Dense(units=1))

# Derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Erken Durdurma
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Eğitimi Başlat
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=16, 
                    validation_data=(x_test, y_test), 
                    callbacks=[early_stop],
                    verbose=1)

# Loss Grafiği
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Eğitim Hatası')
plt.plot(history.history['val_loss'], label='Test/Doğrulama Hatası')
plt.title('Model Eğitim Süreci')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# 8. TAHMİN VE TERS ÖLÇEKLEME (INVERSE TRANSFORM)
# ----------------------------------------------------------------
# Tahmin Yap
y_pred_scaled = model.predict(x_test)

# Ölçeklenmiş veriyi gerçek değerlere (Watt) çevirme
# Özellik sayısı (Feature count) kadar sütunu olan boş matris lazım.
feature_count = train_data.shape[1] # Özellik sayısını buradan alıyoruz (HATA BURADAYDI)

# Tahminler için matris
dummy_pred = np.zeros((len(y_pred_scaled), feature_count))
dummy_pred[:, -1] = y_pred_scaled.flatten() 
y_pred_inverse = sc.inverse_transform(dummy_pred)[:, -1]

# Gerçek değerler için matris
dummy_test = np.zeros((len(y_test), feature_count))
dummy_test[:, -1] = y_test
y_test_inverse = sc.inverse_transform(dummy_test)[:, -1]

# Performans Metrikleri
rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
mae = mean_absolute_error(y_test_inverse, y_pred_inverse)

print("="*30)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE:  {mae:.2f}")
print("="*30)

# 9. SONUÇLARIN GÖRSELLEŞTİRİLMESİ
# ----------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(y_test_inverse, color='red', label='Gerçek Tüketim')
plt.plot(y_pred_inverse, color='blue', label='LSTM Tahmini')
plt.title('Güç Tüketimi Tahmini (Günlük)')
plt.xlabel('Günler (Test Seti)')
plt.ylabel('Tüketim (TotalPowerConsumption)')
plt.legend()

plt.show()
