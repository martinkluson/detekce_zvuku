import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 🗂️ Cesta k adresáři s daty a názvy tříd
DATA_DIR = "zvuky"  # Složka, kde jsou uloženy zvukové soubory
CLASSES = ["alarm", "normalni"]  # Seznam tříd - kategorie zvuků

# 🎵 Funkce pro extrakci MFCC příznaků z audio souboru
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Načtení zvukového souboru
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extrakce MFCC příznaků
    return np.mean(mfcc.T, axis=0)  # Vypočítání průměru MFCC přes časovou osu

# 📝 Inicializace prázdných seznamů pro příznaky (X) a označení tříd (y)
X = []
y = []

# 📊 Výpis počtu souborů v každé třídě
print("📊 Počet souborů podle tříd:")

# 📁 Procházení složek pro každou třídu
for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, cls)  # Určení cesty k dané třídě
    if not os.path.exists(folder):  # Pokud složka neexistuje, vytvoříme ji
        print(f"⚠️ Složka {folder} neexistuje! Vytvářím ji.")
        os.makedirs(folder)  # Vytvoření složky pro danou třídu
    
    count = 0  # Počítadlo pro soubory v aktuální třídě
    # 🗂️ Procházení všech souborů ve složce
    for file in os.listdir(folder):
        if file.endswith(".wav"):  # Zpracováváme pouze soubory s příponou .wav
            file_path = os.path.join(folder, file)
            try:
                features = extract_features(file_path)  # Extrakce příznaků
                X.append(features)  # Přidání příznaků do seznamu X
                y.append(label)  # Přidání označení třídy do seznamu y
                count += 1  # Zvýšení počtu souborů v aktuální třídě
            except Exception as e:
                print(f"❌ Chyba při načítání {file_path}: {e}")  # Chyba při načítání souboru
    
    # Výpis počtu souborů pro aktuální třídu
    print(f"  - {cls}: {count} souborů")

# 🌍 Převedení seznamů na numpy pole pro strojové učení
X = np.array(X)
y = np.array(y)

# 🛑 Kontrola, zda máme dostatek dat
if len(X) < 2:
    print("❌ Nedostatek dat. Potřebujete alespoň 2 zvukové soubory.")
    exit()

# 🔀 Rozdělení dat na trénovací a testovací sady (25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 🔧 Normalizace příznaků pro trénování modelu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Naučení a aplikace na trénovací data
X_test = scaler.transform(X_test)  # Aplikace stejného scaleru na testovací data

# 🧠 Nastavení KNN modelu a počet sousedů (3 nebo méně, podle počtu trénovacích dat)
n_neighbors = min(3, len(X_train))
if n_neighbors < 1:
    print("❌ Nedostatek trénovacích dat pro klasifikaci.")
    exit()

# 👨‍💻 Trénování KNN modelu
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)  # Trénování modelu na trénovacích datech
y_pred = model.predict(X_test)  # Predikce na testovacích datech

# 🧾 Výpis výsledků klasifikace (precision, recall, f1-score)
print("\n=== Výsledky detekce zvuků ===")
print(classification_report(y_test, y_pred, target_names=CLASSES))  # Výpis metrik

# 🎨 Vizualizace MFCC pro jeden vzorek z první třídy
sample_file = os.path.join(DATA_DIR, CLASSES[0], os.listdir(os.path.join(DATA_DIR, CLASSES[0]))[0])
y_sample, sr_sample = librosa.load(sample_file)  # Načtení vzorku
mfcc = librosa.feature.mfcc(y=y_sample, sr=sr_sample, n_mfcc=13)  # Extrakce MFCC příznaků

# 🖼️ Zobrazení MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr_sample)  # Zobrazení MFCC jako spektrální mapy
plt.colorbar()  # Přidání barevné stupnice
plt.title("MFCC vizualizace - " + CLASSES[0])  # Titulek pro graf
plt.tight_layout()  # Optimalizace rozvržení
plt.show()  # Zobrazení grafu
