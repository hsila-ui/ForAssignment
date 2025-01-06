# import tarfile
#tar_path = "/Users/user/PycharmProjects/PythonProject1/Datasets/CTU-13-Dataset.tar.bz2"
#output_dir = "./extracted_files/"

#try:
   # with tarfile.open(tar_path, "r:bz2") as tar:
       # for member in tar:
            #if member.name.endswith(".pcap"):
               # tar.extract(member, path=output_dir)
               # print(f"{member.name} dosyası {output_dir} içine çıkarıldı.")
#except FileNotFoundError:
  #  print(f"Hata: {tar_path} dosyası bulunamadı.")

#from scapy.all import rdpcap
#import pandas as pd
#import numpy as np

# PCAP Dosyasını Okuma ve Veri Çıkarma

#def read_pcap(file_path):
   # packets = rdpcap(file_path)
    #packet_data = []

   # for packet in packets:
       # try:
           # packet_data.append({
               # "timestamp": packet.time if hasattr(packet, "time") else None,
                #"src_ip": packet[0][1].src if packet.haslayer("IP") else None,
                #"dst_ip": packet[0][1].dst if packet.haslayer("IP") else None,
                #"protocol": packet[0][1].proto if packet.haslayer("IP") else None,
                #"length": len(packet)
           # })
       # except IndexError:
            # Paket eksik veya hatalı ise atla
           # continue

    #return pd.DataFrame(packet_data)

# Eksik Veri Tespiti
#def detect_missing_values(data):
   # print("Eksik Veri Sayısı:")
   # print(data.isnull().sum())
    #print("\nEksik Veri Oranı (%):")
    #missing_ratio = data.isnull().sum() / len(data) * 100
    #print(missing_ratio)

# Eksik Verileri Doldurma
#def fill_missing_values(data):
    #data['src_ip'].fillna("0.0.0.0", inplace=True)  # Eksik IP adresleri varsayılan bir değerle doldur
   # data['dst_ip'].fillna("0.0.0.0", inplace=True)
    #data['protocol'].fillna("UNKNOWN", inplace=True)  # Eksik protokoller için 'UNKNOWN'
     #data['length'].fillna(data['length'].mean(), inplace=True)  # Paket uzunluklarını ortalama ile doldur
    #return data

# Sonuçları Kaydetme
#def save_to_csv(data, file_name):
    #data.to_csv(file_name, index=False)
    #print(f"İşlenmiş veri kaydedildi: {file_name}")

# İşleme Adımları
#if __name__ == "__main__":
    # PCAP dosyasının yolu
    #pcap_file_path = "extracted_files/CTU-13-Dataset/13/botnet-capture-20110815-fast-flux-2.pcap"
    # PCAP verisini yükleme
    #print("PCAP dosyası yükleniyor...")
    #pcap_data = read_pcap(pcap_file_path)
    #print("PCAP verisi başarıyla yüklendi.")

    # Eksik veri tespiti
    #print("Eksik veriler kontrol ediliyor...")
    #detect_missing_values(pcap_data)

    # Eksik veriyi doldurma
    #print("Eksik veriler dolduruluyor...")
    #processed_data = fill_missing_values(pcap_data)

    # Sonuçları CSV formatında kaydetme
    #output_file = "processed_pcap_data13.csv"
    #save_to_csv(processed_data, output_file)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import IsolationForest


# CSV dosyasını yükleyin
data = pd.read_csv('processed_pcap_data1.csv')

# Protokol sütununu dönüştürme (eğer mevcutsa)
if 'protocol' in data.columns:
    encoder = LabelEncoder()
    data['protocol'] = encoder.fit_transform(data['protocol'])

# IP adreslerini sayısal değerlere dönüştürme (isteğe bağlı)
data['src_ip'] = data['src_ip'].astype('category').cat.codes
data['dst_ip'] = data['dst_ip'].astype('category').cat.codes







# Timestamp sütununu datetime formatına çevir
data['datetime'] = pd.to_datetime(data['timestamp'], unit='s', errors='coerce')

# Hour ve Minute özelliklerini türet
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute

# Time Interval (zaman farkı) özelliğini türet
data['time_diff'] = data['timestamp'].diff().fillna(0)

# Eksik verileri doldurma
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Sayısal sütunları yeniden belirleyin
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Model giriş verisini belirleme
X = data[numeric_data.columns]


# Sayısal sütunları ölçeklendirme
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Isolation Forest modelini tanımlama
isolation_forest = IsolationForest(contamination=0.1, random_state=42)

# Modeli eğitme
X = data[numeric_data.columns]  # Model yalnızca sayısal sütunlarla çalışır
isolation_forest.fit(X)

# Tahminler ve anomali skorları
data['anomaly_score'] = isolation_forest.decision_function(X)  # Anomali skorları
data['is_anomaly'] = isolation_forest.predict(X)  # Tahminler: 1 (normal), -1 (anormal)

# Anomali ve normal veri sayısını yazdırma
print("Anomali Tespiti Dağılımı:")
print(data['is_anomaly'].value_counts())

# Yüzde olarak dağılım
anomaly_percentage = data['is_anomaly'].value_counts(normalize=True) * 100
print("\nAnomali Dağılımı (%):")
print(anomaly_percentage)







# Tahmin sonuçlarını kontrol et
print("Anomali Tespiti Sonuçları:")
print(data['is_anomaly'].value_counts())

# Anomalileri görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='hour', y='time_diff', hue='is_anomaly', palette={1: 'blue', -1: 'red'})
plt.title('Anomaly Detection: Hour vs Time Difference')
plt.xlabel('Hour')
plt.ylabel('Time Difference')
plt.legend(['Normal', 'Anomaly'])
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['anomaly_score'], bins=50, kde=True)
plt.title('Anomaly Score Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.show()









# Yeni özelliklerin başarıyla oluşturulduğunu kontrol etmek için ilk birkaç satırı görüntüleyin
print(data[['timestamp', 'datetime', 'hour', 'minute', 'time_diff']].head())

# Veri türleri ve eksik değerler hakkında genel bilgi
print("Veri türleri ve eksik değerler:")
print(data.info())

# Eksik değer içeren sütunları tespit edin
print("\nEksik değer içeren sütunlar:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# Sayısal ve sayısal olmayan sütunları ayırın
numeric_data = data.select_dtypes(include=['float64', 'int64'])
non_numeric_data = data.select_dtypes(exclude=['float64', 'int64'])

# Eksik değerleri doldurma
data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())
if not non_numeric_data.empty:
    data[non_numeric_data.columns] = non_numeric_data.fillna(non_numeric_data.mode().iloc[0])

# Eksik değerler tekrar kontrol edilir
print("\nEksik değerler doldurulduktan sonra kontrol:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# Temel istatistikler
print("\nTemel istatistikler:")
print(data.describe())

# Gereksiz sütunları çıkarma (örneğin: id, timestamp, src_ip, dst_ip)
columns_to_drop = ['id', 'timestamp', 'src_ip', 'dst_ip']
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
data = data.drop(columns=existing_columns_to_drop)

# Sayısal sütunları yeniden belirleyin
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Veri dönüşümleri (normalizasyon ve label encoding)
scaler = StandardScaler()
data[numeric_data.columns] = scaler.fit_transform(data[numeric_data.columns])

encoder = LabelEncoder()
if 'protocol' in data.columns:
    data['protocol'] = encoder.fit_transform(data['protocol'])

# Korelasyon Matrisi (Seçili Sütunlarla)
subset_columns = numeric_data.columns[:5]  # İlk 5 sayısal sütunu seçin
subset_correlation_matrix = data[subset_columns].corr()
print("Seçili Sütunlarla Korelasyon Matrisi:")
print(subset_correlation_matrix)

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(10, 6))
sns.heatmap(subset_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Subset Columns)')
plt.show()

# 1. Veri Dağılımının İncelenmesi
# Tüm sayısal sütunlar için histogram
for column in numeric_data.columns:
    plt.figure()
    data[column].hist(bins=30)
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# 2. Aykırı Değer Tespiti
for column in numeric_data.columns:
    plt.figure()
    sns.boxplot(data[column])
    plt.title(f'{column} Outlier Detection')
    plt.show()

# 3. Kategorik Özelliklerin Dağılımı
categorical_columns = data.select_dtypes(include=['object', 'category']).columns
for column in categorical_columns:
    plt.figure()
    data[column].value_counts().plot(kind='bar')
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# 4. Özellikler Arasındaki Korelasyon
if not numeric_data.empty:
    correlation_matrix = data[numeric_data.columns].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

# 5. Çoklu Korelasyon Analizi (Pairplot)
if len(numeric_data.columns) > 1:
    sns.pairplot(data[numeric_data.columns])
    plt.show()

# 6. Zaman Serisi Analizi (Varsa)
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    if not data['timestamp'].isna().all():
        plt.figure(figsize=(10, 6))
        data.set_index('timestamp')['length'].plot()
        plt.title('Packet Length Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Packet Length')
        plt.show()

# 7. Sınıf Dağılımı Analizi (Varsa)
if 'label' in data.columns:  # 'label' veya benzeri bir hedef değişken kontrolü
    plt.figure()
    data['label'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

# Anormal veri noktalarını seçme
#anomalies = data[data['is_anomaly'] == -1]

# İlk 10 anomaliyi yazdır
#print("Tespit Edilen Anomaliler (İlk 10):")
#print(anomalies.head(10))


# Anomalileri bir CSV dosyasına kaydet
#anomalies.to_csv('anomalies_detected9.csv', index=False)
#print("Anomaliler anomalies_detected.csv dosyasına kaydedildi.")