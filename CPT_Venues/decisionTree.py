import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ast
from nltk.sentiment import SentimentIntensityAnalyzer
from joblib import Parallel, delayed

reviews_train = pd.read_csv("reviewsTrainToronto.csv")
reviews_test = pd.read_csv("reviewsTestToronto.csv")
X_train_raw = pd.read_csv("X_trainToronto.csv")
X_test_raw = pd.read_csv("X_testToronto.csv")
sia = SentimentIntensityAnalyzer()


def get_sentiment_score(text):
    return sia.polarity_scores(str(text))['compound']


def compute_sentiment_parallel(texts, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(delayed(get_sentiment_score)(t) for t in texts)


reviews_train['sentiment'] = compute_sentiment_parallel(reviews_train['text'])
reviews_test['sentiment'] = compute_sentiment_parallel(reviews_test['text'])


def count_attributes(attr_str):
    try:
        attr_dict = ast.literal_eval(attr_str)
        if isinstance(attr_dict, dict):
            return len(attr_dict)
        else:
            return 0
    except:
        return 0


X_train_raw['attributes_count'] = X_train_raw['attributes'].apply(count_attributes)
X_test_raw['attributes_count'] = X_test_raw['attributes'].apply(count_attributes)
for df in [reviews_train, reviews_test]:
    df['review_length'] = df['text'].astype(str).apply(len)


def agregar_reviews(df):
    return df.groupby('business_id').agg({
        'useful': 'sum',
        'funny': 'sum',
        'cool': 'sum',
        'review_length': 'mean',
        'sentiment': 'mean'
    }).rename(columns={
        'useful': 'useful_sum',
        'funny': 'funny_sum',
        'cool': 'cool_sum',
        'review_length': 'review_length_mean',
        'sentiment': 'sentiment_mean'
    }).reset_index()


agg_train = agregar_reviews(reviews_train)
agg_test = agregar_reviews(reviews_test)
train = pd.merge(X_train_raw, agg_train, on='business_id', how='inner')
test = pd.merge(X_test_raw, agg_test, on='business_id', how='inner')
n_clusters = 10
coords_train = train[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
train['location_cluster'] = kmeans.fit_predict(coords_train)
coords_test = test[['latitude', 'longitude']]
test['location_cluster'] = kmeans.predict(coords_test)
train = pd.get_dummies(train, columns=['location_cluster'], prefix='locclust')
test = pd.get_dummies(test, columns=['location_cluster'], prefix='locclust')
train, test = train.align(test, join='left', axis=1, fill_value=0)
funny_thresh = train['funny_sum'].quantile(0.90)
useful_thresh = train['useful_sum'].quantile(0.10)
train = train[~((train['funny_sum'] > funny_thresh) & (train['useful_sum'] < useful_thresh))]
z_scores = (train['review_length_mean'] - train['review_length_mean'].mean()) / train['review_length_mean'].std()
train = train[np.abs(z_scores) < 3]
base_features = ['review_count', 'useful_sum', 'cool_sum', 'funny_sum', 'review_length_mean', 'attributes_count',
                 'sentiment_mean']
cluster_features = [col for col in train.columns if col.startswith('locclust_')]
features = base_features + cluster_features
X = train[features]
y = train['destaque']
X_test = test[features]
print("Distribuição do target (destaque):")
print(y.value_counts(normalize=True))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
)
tree_model.fit(X_train, y_train)
y_val_pred = tree_model.predict(X_val)
cond_val = (X_val['funny_sum'] > funny_thresh) & (X_val['useful_sum'] < useful_thresh)
y_val_pred[cond_val.values] = 0
accuracy = accuracy_score(y_val, y_val_pred)
print(f'\nAcurácia no conjunto de validação (Decision Tree) com regra aplicada: {accuracy:.2%}')
print("Classification Report (validação com regra):")
print(classification_report(y_val, y_val_pred))
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matriz de Confusão - Decision Tree (Validação com regra)")
plt.show()
cv_scores = cross_val_score(tree_model, X, y, cv=5, scoring='accuracy')
print(f'Acurácia média (Decision Tree CV-5): {cv_scores.mean():.2%} ± {cv_scores.std():.2%}')
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)
y_val_pred_log = log_model.predict(X_val)
y_val_pred_log[cond_val.values] = 0
accuracy_log = accuracy_score(y_val, y_val_pred_log)
print(f'\nAcurácia no conjunto de validação (Logistic Regression) com regra aplicada: {accuracy_log:.2%}')
print("Classification Report (validação - Logistic Regression com regra):")
print(classification_report(y_val, y_val_pred_log))
cm_log = confusion_matrix(y_val, y_val_pred_log)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp_log.plot()
plt.title("Matriz de Confusão - Logistic Regression (Validação com regra)")
plt.show()
cv_scores_log = cross_val_score(log_model, X, y, cv=5, scoring='accuracy')
print(f'Acurácia média (Logistic Regression CV-5): {cv_scores_log.mean():.2%} ± {cv_scores_log.std():.2%}')
tree_model.fit(X, y)
y_pred_tree = tree_model.predict(X_test)
cond_test = (test['funny_sum'] > funny_thresh) & (test['useful_sum'] < useful_thresh)
y_pred_tree[cond_test.values] = 0
submission = test[['business_id']].copy()
submission['destaque'] = y_pred_tree
submission.to_csv("submission.csv", index=False)
y_pred_train = tree_model.predict(X)
print(classification_report(y, y_pred_train))
importances = pd.Series(tree_model.feature_importances_, index=features)
importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importance')
plt.tight_layout()
plt.show()
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=features, class_names=["0", "1"], filled=True)
plt.show()
