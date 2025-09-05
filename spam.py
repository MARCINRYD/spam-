
import numpy as np
import pandas as pd
import string
import nltk
import itertools
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import re

# Pobieranie niezbędnych zasobów NLTK
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Open Multilingual WordNet
except:
    print("Uwaga: Problem z pobraniem zasobów NLTK. Kontynuowanie...")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ścieżka do pliku z danymi
sciezka_do_pliku = r'C:\Users\MSI\Desktop\spam\spam.csv'

# Wczytanie danych
spam_dataset = pd.read_csv(sciezka_do_pliku, 
                          encoding="ISO-8859-1", 
                          usecols=[0, 1], 
                          names=['Spam', 'Text'],
                          skiprows=1)

# Konwersja etykiet (ham -> 0, spam -> 1)
spam_dataset['Spam'] = spam_dataset['Spam'].map({'ham': 0, 'spam': 1})

# Funkcja do usuwania interpunkcji
def remove_puncation(text):
    if not isinstance(text, str):
        return ""
    cleaned = ''.join([word for word in text if word not in string.punctuation])
    return cleaned

# Dodanie kolumny z oczyszczonym tekstem (bez interpunkcji)
spam_dataset['Cleaned_Text'] = spam_dataset['Text'].apply(lambda x: remove_puncation(x))

# Prostsza funkcja tokenizacji bez użycia word_tokenize
def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    # Usunięcie wielkich liter
    clean_text = text.lower()
    # Prosta tokenizacja przez podział na słowa
    tokens = re.findall(r'\b\w+\b', clean_text)
    return tokens

# Dodanie kolumny z tokenizowanym tekstem
spam_dataset['Tokenized_Text'] = spam_dataset['Cleaned_Text'].apply(lambda x: simple_tokenize(x))

# Pobranie listy stop words
stopwords_list = nltk.corpus.stopwords.words("english")

# Funkcja do usuwania stop words
def remove_stopwords(text):
    without_stopwords = [word for word in text if word not in stopwords_list]
    return without_stopwords

# Dodanie kolumny z tekstem bez stop words
spam_dataset['WithoutStop_Text'] = spam_dataset['Tokenized_Text'].apply(lambda x: remove_stopwords(x))

# Inicjalizacja lematyzera
lemmatizer = nltk.WordNetLemmatizer()

# Funkcja do lematyzacji
def lemmatizing(text):
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized_words

# Dodanie kolumny z lematyzacją
spam_dataset['Lemmatized_Text'] = spam_dataset['WithoutStop_Text'].apply(lambda x: lemmatizing(x))

# Przygotowanie danych do modelowania
spam_dataset['Processed_Text'] = spam_dataset['Lemmatized_Text'].apply(lambda x: ' '.join(x))

# Wektoryzacja TF-IDF
print("Wektoryzacja TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(spam_dataset['Processed_Text'])
y = spam_dataset['Spam']

print(f"Kształt danych przed podziałem: {X.shape}")

# Podział na zbiór treningowy i testowy z stratyfikacją
print("\nPodział na zbiór treningowy i testowy...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Kształt zbioru treningowego: {X_train.shape}")
print(f"Kształt zbioru testowego: {X_test.shape}")

# Trenowanie początkowego modelu Random Forest
print("\nTrenowanie początkowego modelu Random Forest...")
rf_initial = RandomForestClassifier(n_estimators=100, random_state=42)
rf_initial.fit(X_train, y_train)

# Ocena początkowego modelu
y_pred_initial = rf_initial.predict(X_test)
accuracy_initial = accuracy_score(y_test, y_pred_initial)
print(f"Dokładność początkowego modelu: {accuracy_initial:.4f}")

# Wyciągnięcie ważności cech
feature_importances = rf_initial.feature_importances_
feature_names = tfidf.get_feature_names_out()

# Tworzenie DataFrame z ważnościami cech
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

# Selekcja cech z ważnością > 0.001
selected_features = importance_df[importance_df['importance'] > 0.001]['feature'].values
print(f"\nLiczba wszystkich cech: {len(feature_names)}")
print(f"Liczba wyselekcjonowanych cech: {len(selected_features)}")

# Indeksy wyselekcjonowanych cech
selected_indices = [i for i, feature in enumerate(feature_names) if feature in selected_features]

# Redukcja zbiorów danych do wyselekcjonowanych cech
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

print(f"Kształt zbioru treningowego po selekcji: {X_train_selected.shape}")
print(f"Kształt zbioru testowego po selekcji: {X_test_selected.shape}")

# GridSearch dla optymalizacji hiperparametrów
print("\nPrzeprowadzanie GridSearch dla optymalizacji hiperparametrów...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Użycie StratifiedKFold dla walidacji krzyżowej
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch z walidacją krzyżową
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_selected, y_train)

# Najlepsze parametry i model
print(f"\nNajlepsze parametry: {grid_search.best_params_}")
print(f"Najlepszy wynik walidacji krzyżowej: {grid_search.best_score_:.4f}")

# Ocena najlepszego modelu na zbiorze testowym
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_selected)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Dokładność najlepszego modelu na zbiorze testowym: {accuracy_best:.4f}")

# Raport klasyfikacji
print("\nRaport klasyfikacji dla najlepszego modelu:")
print(classification_report(y_test, y_pred_best))

# Macierz pomyłek
print("Macierz pomyłek:")
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywistość')
plt.title('Macierz pomyłek dla modelu Random Forest')
plt.show()

# Najważniejsze cechy po selekcji
top_features = importance_df.head(20)
plt.figure(figsize=(10, 8))
plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
plt.xlabel('Ważność cechy')
plt.title('20 najważniejszych cech w modelu')
plt.tight_layout()
plt.show()

print("\nSkrypt zakończony pomyślnie!")
