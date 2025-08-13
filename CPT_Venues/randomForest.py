# !pip install swifter nltk scikit-learn xgboost

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import re
import swifter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def preprocess_text(text):
  if text is None:
    return ""

  # Converter para letras minusculas
  text = text.lower()

  # Remover caracteres especiais e links
  text = re.sub(r"http\S+|www\S+|@\S+", "", text)
  text = re.sub(r"[^a-zA-Z\s]", "", text)

  # Tokenizacao e remocao de stopwords
  tokens = text.split()
  tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

  return " ".join(tokens)

def prepare_reviews_dataset(df_reviews):
  # Quantidade de caracteres
  df_reviews["text_length"] = df_reviews["text"].apply(len)

  # Features de tempo das reviews
  df_reviews['date'] = pd.to_datetime(df_reviews['date'])
  df_reviews['review_age_days'] = (pd.to_datetime('today') - df_reviews['date']).dt.days

  # Limpar texto reviews
  df_reviews["cleaned_text"] = df_reviews["text"].swifter.apply(preprocess_text)
  df_reviews["sentiment"] = df_reviews["cleaned_text"].swifter.apply(lambda x: sia.polarity_scores(x)["compound"])

  # Contar soma e media de reacoes por estabelecimento
  grouped_reviews = df_reviews.groupby('business_id').agg({
      'useful': ['mean', 'sum'],
      'funny': ['mean', 'sum'],
      'cool': ['mean', 'sum'],
      'text_length': ['mean'],
      'review_age_days': ['mean'],
      'sentiment': ['mean']
      }).reset_index()
  grouped_reviews.columns = ['business_id', 'useful_mean', 'useful_sum', 'funny_mean', 'funny_sum', 'cool_mean', 'cool_sum', 'text_length_mean', 'review_age_days_mean', 'sentiment_mean']

  return grouped_reviews

def prepare_business_dataset(df_business, df_reviews):
  # Lidar com valores ausentes
  df_business.fillna({'review_count': 0, 'useful': 0, 'funny': 0, 'cool': 0}, inplace=True)
  df_business['categories'] = df_business['categories'].fillna('')
  df_business["attributes"] = df_business["attributes"].fillna("{}")  # Se for JSON vazio
  df_business["hours"] = df_business["hours"].fillna("{}")

  # Extrair features de texto das categorias
  # vectorizer = TfidfVectorizer(max_features=100)
  # categories_features = vectorizer.fit_transform(df_business['categories'])

  # Codificar features categoricas
  # df_business["categories"] = df_business["categories"].str.split(", ")
  # mlb = MultiLabelBinarizer()
  # categories_encoded = pd.DataFrame(mlb.fit_transform(df_business["categories"]), columns=mlb.classes_)
  # df_business = pd.concat([df_business, categories_encoded], axis=1)

  # Determinar se localizacao eh central
  df_business["downtown"] = ((df_business['latitude'].between(43.6, 43.7)) & (df_business['longitude'].between(-79.4, -79.3))).astype(int)

  # Normalizar review_count
  df_business["log_review_count"] = np.log1p(df_business["review_count"])

  # Preparar dataset das reviews
  prepared_df_reviews = prepare_reviews_dataset(df_reviews)

  # Agregar dados obtidos
  df_business = pd.merge(df_business, prepared_df_reviews, on='business_id', how='left')

  return df_business

# TREINO
business_train = pd.read_csv('X_trainToronto.csv')
reviews_train = pd.read_csv('reviewsTrainToronto.csv')

# Preparar dados
merged_data = prepare_business_dataset(business_train, reviews_train)
X = merged_data[['review_count', 'useful_mean', 'useful_sum', 'funny_mean', 'funny_sum', 'cool_mean', 'cool_sum', 'text_length_mean', 'review_age_days_mean', 'sentiment_mean', 'downtown', 'log_review_count']]
y = merged_data['destaque']

# Normalizar dados
feature_names = X.columns
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir em treino e validacao
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Computar pesos (desbalanceamento)
weights = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}

# Random Forest - balanceada
balanced_rf = RandomForestClassifier(class_weight=class_weights, n_estimators=100)
balanced_rf_scores = cross_val_score(balanced_rf, X_train, y_train, cv=5)
print(f"Random Forest: {balanced_rf_scores.mean():.4f} ± {balanced_rf_scores.std():.4f}")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(balanced_rf, param_grid, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# Melhores parametros encontrados
best_rf = grid_search.best_estimator_
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor score: {grid_search.best_score_:.4f}")

# Previsões no conjunto de validação
y_pred = best_rf.predict(X_val)

# Metricas de avaliação
print("Relatório de Classificação:")
print(classification_report(y_val, y_pred))

print("\nAcurácia:", accuracy_score(y_val, y_pred))
print("F1-Score:", f1_score(y_val, y_pred))

# TESTE
business_test = pd.read_csv('X_testToronto.csv')
reviews_test = pd.read_csv('reviewsTestToronto.csv')

# Preparar dados
test_data = prepare_business_dataset(business_test, reviews_test)
X_test = test_data[['review_count', 'useful_mean', 'useful_sum', 'funny_mean', 'funny_sum', 'cool_mean', 'cool_sum', 'text_length_mean', 'review_age_days_mean', 'sentiment_mean', 'downtown', 'log_review_count']]

# Normalizar dados
X_test = scaler.transform(X_test)

# Prever
test_data['destaque'] = best_rf.predict(X_test)

# Salvar dados para arquivo CSV
output_df = test_data[['business_id', 'destaque']]
output_df.to_csv('sampleResposta.csv', index=False)
print("Arquivo salvo: sampleResposta.csv")
print(output_df.head())
print(output_df.info())
print(output_df['destaque'].value_counts())
