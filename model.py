import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.utils import shuffle
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
warnings.filterwarnings('ignore')
data = pd.read_csv("phishing_site_urls.csv")
df_shuffled = shuffle(data, random_state=42)
data_size = 5000
df_used = df_shuffled[:data_size].copy()
df_used.replace({'good':0, 'bad':1}, inplace=True)
X = df_used[['URL']].copy()
y = df_used.Label.copy()
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")
cv = CountVectorizer()
rfc = LogisticRegression()
training_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def prepare_data(X) :
    X['text_tokenized'] = X.URL.map(lambda t: tokenizer.tokenize(t))
    X['text_stemmed'] = X.text_tokenized.map(lambda t: [stemmer.stem(word) for word in t])
    X['text_sent'] = X.text_stemmed.map(lambda t: ' '.join(t))
    features = cv.fit_transform(X.text_sent)
    return X, features
X, features = prepare_data(X)
for p in training_sizes:
    trainX, testX, trainY, testY = train_test_split(X, y, stratify=y,test_size=1-p, random_state=42)
    trainX = trainX.reset_index(drop=True)
    trainY = trainY.reset_index(drop=True)
    rfc.fit(features[trainX['text_sent'].index.values], trainY)

# Save the trained model
joblib.dump(rfc, "phishing_model.pkl")
