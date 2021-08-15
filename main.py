# Imports

import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score

vectorizer_review = TfidfVectorizer(strip_accents='ascii', min_df=1)
clf_svc_review = SVC(kernel='linear')

# Extracting Data
data = []
categories = ['Books', 'Cars', 'Clothing', 'Electronics', 'Garden', 'Health', 'Movies', 'Pets', 'Sports', 'VideoGames']

sentiment_list = {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}

for category in categories:
    with open(f'./Data/{category}.json', 'r') as f:
        for line in f:
            review = json.loads(line)
            data.append(
                {'category': category, 'text': review['reviewText'], 'sentiment': sentiment_list[review['overall']]})

clf_data_review = dict()

for review in data:
    clf_data_review[review['text']] = review['sentiment']


# Equally distributing POSITIVE and NEGATIVE reviews
def filter_dict(d, func):
    newDict = {}
    for key, value in d.items():
        if func(key, value):
            newDict[key] = value
    return newDict


neg_dict = filter_dict(clf_data_review, lambda x, y: y == "Negative")
pos_dict = filter_dict(clf_data_review, lambda x, y: y == "Positive")

pos_dict = dict(list(pos_dict.items())[:len(neg_dict)])
neg_dict.update(pos_dict)
clf_data_review = neg_dict

data_list = list(clf_data_review.items())
np.random.shuffle(data_list)
clf_data_review = dict(data_list)

# Seperating Data
# Using train_test_split

X_train, X_test = train_test_split(list(clf_data_review.keys()), test_size=0.2, random_state=69)
y_train_review, y_test_review = train_test_split(list(clf_data_review.values()), test_size=0.2, random_state=69)

# Converting string data into numerical data
# Using TfidfVectorizer

X_train_review = vectorizer_review.fit_transform(X_train)
X_test_review = vectorizer_review.transform(X_test)

# Using Classifier Models
# Support Vector Classifier

clf_svc_review.fit(X_train_review, y_train_review)

# Checking score

score = clf_svc_review.score(X_test_review, y_test_review)
f1_score = f1_score(y_test_review, clf_svc_review.predict(X_test_review), average=None)

# Manual testing

text = vectorizer_review.transform([input("Review: ")])
print(clf_svc_review.predict(text))
