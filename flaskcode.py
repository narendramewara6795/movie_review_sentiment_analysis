import maincode
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analysis', methods=['GET', 'POST'])
def sentimentAnalysis():

    algorithm = request.form.get('algorithm')
    if algorithm == 'lr':
        classifier = maincode.linearRegressor
        algorithm = 'Linear Regression'
    # elif algorithm == 'lgr':
    #     classifier = maincode.logisticRegressor
    #     algorithm = 'Logistic Regression'
    elif algorithm == 'rf':
        classifier = maincode.rfClassifier
        algorithm = 'Random Forest Classification'
    elif algorithm == 'mnb':
        classifier = maincode.mnbClassifier
        algorithm = 'Multinomial Naive Bayes Classification'

    accuracy = maincode.find_accuracy(classifier)
    review  = request.form.get('review')
    sentiment = maincode.find_sentiment(classifier,review)

    return render_template('result.html', algorithm=algorithm, accuracy=accuracy,sentiment=sentiment)

app.run(host='0.0.0.0', port=5000)
