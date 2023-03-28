# K - Nearest Neighbours Modeli 
# Hasta Tahlil Tahmin Uygulaması

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Outcome = 1 Şeker hastası
# Outcome = 0 Sağlıklı
#************************

data = pd.read_csv("diabetes.csv")  # Veri seti sisteme yüklendi
print(data.head)

#*****************
# Şimdilik sadece gluocose a bakarak örnek bir çizim yapalım.
# Programın geri kalanında sadece gluocose a değil tüm verilere bakarak tahmin yapacağız.

sekerHastalari = data[data.Outcome == 1 ]   #Sağlıklı insanları bir değişkene atadık.
saglikliİnsanlar = data[data.Outcome == 0]  #Hasta insanları bir değişkene atadık.

plt.scatter(saglikliİnsanlar.Age, saglikliİnsanlar.Glucose, color="green", label="Sağlıklı", alpha= 0.4)
plt.scatter(sekerHastalari.Age, sekerHastalari.Glucose, color="red", label="Diabet Hastaları", alpha= 0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()

#*******************

# Şimdi x ve y verilerini yani girdi ve çıktı verilerini belirleyelim.

y = data.Outcome.values     #Çıktı yani sonuç değerleri.
x_ham_veri = data.drop(["Outcome"], axis=1)     #Tüm dataset içinde outcome sütununu sildik ve kalan verileri girdi verisi yaptık

#Şimdi normalization (x_ham_veri içindeki değerleri KNN modelinin anlayacağı 0 ve 1 değerlerine dönüştürme) yapacağız
#Eğer normalization yapmazsak çıkan sonuç rakamları çok yüksek olur ve model yanılabilir.

x = (x_ham_veri - np.min(x_ham_veri)) / (np.max(x_ham_veri)-np.min(x_ham_veri))    #Normalization işlemi

#Normalization öncesi veri.
print("Normalization öncesi veri: \n")
print(x_ham_veri.head())

#Normalization sonrası veri.
print("Normalization sonrası veriler: \n")
print(x.head())

#Train data ile test datamızı ayıralım.
#Train data modelimizin eğitimi için kullanıcak test datamız ise sistemimizi test etmek için kullanılacak.

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

#Train ve test verilerimizi oluşturduk. Şimdi KNN modelimizi kuruyoruz.

knn = KNeighborsClassifier(n_neighbors= 7) # n_neighbors = k = model en yakın kaç komşuya bakarak hesaplama yapsın
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
print("k=3 için Test verilerimizi doğrulama testi sonucu : ", knn.score(X_test,y_test))

test_verisi = X_test.iloc[21]       #21. indexteki hastanın diabet olma ihtimalini hesapladık
tahmin = knn.predict([test_verisi])

print("Diabet Olma İhtimali : ", tahmin )
#Yukarıda K değerini kafamıza göre belirledik. 3 Verdiğimizde 0.7857142857142857 oranında bir doğruluk altık.
#Şimdi ise K değerimizi kaç vermemiz gerekliğini belirleyeceğiz.


sayac = 1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(X_train,y_train)
    print(sayac," ", "Doğruluk Oranı: %", knn_yeni.score(X_test, y_test))
    sayac += 1

#Yukarıdaki for döngüsü bize k değerinin 7 olduğu durumda en yüksek doğruluk oranı olan 0.8116883116883117 oranını elde ettiğimizi gösteriyor.

