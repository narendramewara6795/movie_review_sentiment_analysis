import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer, SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score

file = open("./dataset/contraction_dict.pkl","rb")
contrations_dictionary = pickle.load(file)

# train = pd.read_csv("./dataset/train.csv")
# test = pd.read_csv("./dataset/test.csv")

movie_dataset = pd.read_csv("./dataset/movie_dataset.csv")
train = movie_dataset.iloc[:40000]
test = movie_dataset.iloc[40000:]

train_review = train["review"]
test_review = test["review"]

lencoder  = LabelEncoder()
train_sentiment = lencoder.fit_transform(train["sentiment"])
test_sentiment = lencoder.fit_transform(test["sentiment"])

# change the data types from list of objects to list np.int8
train_sentiment = np.array(train_sentiment,dtype=np.int8)
test_sentiment = np.array(test_sentiment,dtype=np.int8)

print("---------------------preprocessing--------------------------")
contraction_word_count = 0
def preprocessing(review,removestopwords = False,lemmatization = True):
    # ------------expanding contration word from review-------------
    review = review.lower()
    tokens = review.split(" ")
    decontracted_word = []
    global contraction_word_count
    for word in tokens:
        if word in contrations_dictionary.keys():
            decontracted_word.append(contrations_dictionary.get(word))
            contraction_word_count += 1
        else:
            decontracted_word.append(word)
    review = " ".join(decontracted_word)

    # review = BeautifulSoup(review, "html.parser")
    # review = re.sub("[^a-z]", " ", review.get_text())

    # ------------removing html tags----------------------------------
    html_tags = re.compile('<.*?>')
    review = re.sub(html_tags, " ", review)

    # ------------removing punctuation----------------------------------
    review = re.sub('[^a-z]', " ", review)

    #-------------tokenization----------------------------------------
    tokens = word_tokenize(review)

    # ------------removing stopwords-----------------------
    if removestopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

    # ----------- lemmatization ----------------------------------
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # ----------- stemming ----------------------------------
    else:
        stemmer = SnowballStemmer('english')
        tokens = [stemmer.stem(word) for word in tokens]

    review = " ".join(tokens)
    return review


cleaned_train_review = []
cleaned_test_review = []
for index in range(len(train_review)):
    cleaned_train_review.append(preprocessing(train_review[index],True,True))
for index in range(len(test_review)):
    cleaned_test_review.append(preprocessing(test_review[40000 + index],True,True))
print("contraction word count : {}".format(contraction_word_count))

#------------------Vectorization------------------------------------------------------------
print("--------------------Vectorization-----------------------")
# vectorizer = CountVectorizer(binary= False,max_features=10000)
vectorizer = TfidfVectorizer(binary=False,max_features=10000,ngram_range=(2,3))
vectorizer.fit(cleaned_train_review)
train_features = vectorizer.transform(cleaned_train_review)
test_features = vectorizer.transform(cleaned_test_review)

def linearRegressionClassification():
    print("Linear Regression------------------------------------------------")
    lr = LinearRegression()
    lr.fit(train_features, train_sentiment)
    return lr

def logisticRegressionClassification():
    print("Logistic Regression----------------------------------------------")
    lr = LogisticRegression()
    lr.fit(train_features,train_sentiment)
    return lr

def randomForestClassification():
    print("Random forest classification-------------------------------------")
    rfc = RandomForestClassifier(n_estimators= 100)
    rfc.fit(train_features, train_sentiment)
    return rfc

def mnbClassification():
    print("Multinomial Naive Bayes classification---------------------------")
    mnb = MultinomialNB()
    mnb.fit(train_features,train_sentiment)
    return mnb


linearRegressor = linearRegressionClassification()
# logisticRegressor = logisticRegressionClassification()
rfClassifier = randomForestClassification()
mnbClassifier = mnbClassification()

def find_accuracy(classifier):
    if classifier == linearRegressor:
        predictions = classifier.predict(test_features)
        for index in range(len(predictions)):
            if predictions[index] >= 0.5:
                predictions[index] = 1
            else:
                predictions[index] = 0
    else:
        predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_sentiment, predictions)
    print(f"Accuracy : {accuracy*100:.2f}%")
    return accuracy

def find_sentiment(classifier,review):
    preprocessed_review = [preprocessing(review, True, True)]
    print(preprocessed_review)
    x_test = vectorizer.transform(preprocessed_review)

    if classifier == linearRegressor:
        sentiment = classifier.predict(x_test)
        for index in range(len(sentiment)):
            if sentiment[index] >= 0.5:
                sentiment[index] = 1
            else:
                sentiment[index] = 0
    else:
        sentiment = classifier.predict(x_test)

    if sentiment == 1:
        sentiment = 'pos'
    else:
        sentiment = 'neg'

    print(f"sentiment : {sentiment}")
    return sentiment







