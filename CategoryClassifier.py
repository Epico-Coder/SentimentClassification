# Imports

import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

vectorizer_review = TfidfVectorizer(strip_accents='ascii', min_df=1)
vectorizer_category = TfidfVectorizer(strip_accents='ascii', min_df=1)
clf_svc_category = LinearSVC()

# Extracting Data
data = []
categories = ['Books', 'Cars', 'Clothing', 'Electronics', 'Garden', 'Health', 'Movies', 'Pets', 'Sports', 'VideoGames']
categories2 = ['Books']
sentiment_list = {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}

for category in categories:
    with open(f'./Data/{category}.json', 'r') as f:
        for line in f:
            review = json.loads(line)
            data.append(
                {'category': category, 'text': review['reviewText'], 'sentiment': sentiment_list[review['overall']]})

clf_data_category = dict()

for review in data:
    clf_data_category[review['text']] = review['category']

# Shuffling Data

data_list = list(clf_data_category.items())
np.random.shuffle(data_list)
clf_data_category = dict(data_list)

# Seperating Data
# Using train_test_split

X_train, X_test = train_test_split(list(clf_data_category.keys()), test_size=0.1, random_state=69)
y_train_category, y_test_category = train_test_split(list(clf_data_category.values()), test_size=0.1, random_state=69)

# Converting string data into numerical data
# Using TfidfVectorizer

X_train_category = vectorizer_category.fit_transform(X_train)
X_test_category = vectorizer_category.transform(X_test)

# Using Classifier Models
# Support Vector Classifier

clf_svc_category.fit(X_train_category, y_train_category)

# Checking scores

score = clf_svc_category.score(X_test_category, y_test_category)
f1_score_ = f1_score(y_test_category, clf_svc_category.predict(X_test_category), average=None)

# Manual testing

text = vectorizer_category.transform([input("Review: ")])
print(clf_svc_category.predict(text))
