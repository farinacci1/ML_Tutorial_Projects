# project lib dependencies
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from warnings import simplefilter
# project specs
print('python: {}'.format(sys.version))
print('NLTK: {}'.format(nltk.__version__))
print('scikit-learn: {}'.format(sklearn.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print ('UCI ML-DATASET: {}'.format('https://archive.ics.uci.edu/ml/datasets/sms+spam+collection'))
print('Regex Expressions: {}'.format('http://regexlib.com/Search.aspx?'))
print('Following Tutorial: {}'.format('https://www.youtube.com/watch?v=G4UVJoGFAv0'))
# load data datasets
DATASET = pd.read_csv('SMSSpamCollection',header = None, encoding='utf-8', sep ='\t')
#dataset soecs
print(DATASET.info())
print(DATASET.head())
# data distribution
print(DATASET[0].value_counts())
#preprocessing
encoder = LabelEncoder()
Y = encoder.fit_transform(DATASET[0])
#print first 10 rows in dataset
#print(DATASET[0][:10])
#for first 10 rows print spam or ham
print(Y[:10])# 1 = SPAM 0 = ham
#store all textmessgaes
textMessages = DATASET[1]
#print first 10 messages
#print(textMessages[:10])
#preprocess textMessages
processed = textMessages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddr')#detect emails
processed = processed.str.replace(r'^http[s]{0,1}\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{0,3}(/\S+)?$','webaddr')#detect web addresses
processed =  processed.str.replace(r'؋|ƒ|₡|£|€|₼|\$','currencySymbol')#detect currency symbols
processed =  processed.str.replace(r'^([0-9]( |-)?)?(\(?[0-9]{3}\)?|[0-9]{3})( |-)?([0-9]{3}( |-)?[0-9]{4}|[a-zA-Z0-9]{7})$','phoneNum')#phone numbers
processed =  processed.str.replace(r'\s+',' ')#spaces
processed =  processed.str.replace(r'[^\w\d\s]',' ')#punctuation
processed =  processed.str.replace(r'^\s+|\s+?$','')#remove leading and trailing spaces
processed= processed.str.lower()
#print(processed)
# remove stop words + shorten words to stem
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
ps = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split() ))
#print(processed)
word_tokens = []
for message in processed:
    words = word_tokenize(message)
    for char in words:
        word_tokens.append(char)
word_tokens = nltk.FreqDist(word_tokens)
print('num words {}'.format(len(word_tokens)))
print('most common {}'.format(word_tokens.most_common(15)))
wordFeatures = list(word_tokens.keys())[:1500]

def featureFinder(msg):
    words = word_tokenize(msg)
    features = {}
    for word in wordFeatures:
        features[word] = (word in words)
    return features
###### Start test##############
#features = featureFinder(processed[0])
#for key, value in features.items():
#    if value == True:
#        print (key)
###### End Text test##############
messages = list(zip(processed, Y))

seed = 1
np.random.seed = seed
np.random.shuffle(messages)
#find features for all textMessages
featuresets = [(featureFinder(text),label) for (text,label) in messages]
training,test = model_selection.train_test_split(featuresets,test_size =0.25,random_state = seed)

print('training: {}'.format(len(training)))
print('testing: {}'.format(len(test)))

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)
accuracy = nltk.classify.accuracy(model, test)*100
print("SVC Accuracy: {}".format(accuracy))
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
#train models
models = list(zip(names, classifiers))
for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, test)*100
    print("{} Accuracy: {}".format(name, accuracy))
models = list(zip(names, classifiers))
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, test)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))
# predict
txt_features, labels = list(zip(*test))

prediction = nltk_ensemble.classify_many(txt_features)
print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])
