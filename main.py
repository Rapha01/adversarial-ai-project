import pandas
import csv
import re
import nltk
import sklearn
import torch
import random

porterStemmer = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(10000, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

def sanitizeString(msg):
    # Remove non-words
    sanitizedMsg = re.sub(r'[^a-zA-Z]',' ',msg) 
    # Split words and remove special characters
    sanitizedMsg = nltk.word_tokenize(sanitizedMsg)
    # Stemming (and lower-casing)
    for i in range(len(sanitizedMsg)):
        sanitizedMsg[i] = (porterStemmer.stem(sanitizedMsg[i]))
    # Lemmatizing
    for i in range(len(sanitizedMsg)):
        sanitizedMsg[i] = (lemmatizer.lemmatize(sanitizedMsg[i]))
    sanitizedMsg = ' '.join(sanitizedMsg)

    return sanitizedMsg

def loadAndSanitizeData():
    dataRaw = []
    with open("./data.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            dataRaw.append(row)

    dataRaw = dataRaw[1:]

    data = []
    for d in dataRaw:
        el = []
        el.append(sanitizeString(d[1]))
        if (d[0] == 'ham'):
            el.append(0)
        else:
            el.append(1)
        data.append(el)

    return data

def extractVocabulary(stringArray):
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=10000, stop_words='english')
    cv.fit(stringArray)
    vocabulary = {}
    index = 0
    for feature in cv.get_feature_names_out():
        vocabulary[feature] = index
        index = index + 1
    
    return vocabulary

def splitData(data):
    random.shuffle(data)
    net_data = ''
    test_data = ''


    normal_net_data = ''
    poisoned_net_data = ''

    normal_test_data = ''
    poisoned_test_data = '' 

    return normal_test_data, poisoned_test_data, normal_net_data, poisoned_net_data

def trainModel(trainingdata,vocabulary):
    inputList = [i[0] for i in trainingdata]
    outputList = [i[1] for i in trainingdata]

    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=10000, stop_words='english', vocabulary=vocabulary)
    cv.fit(inputList)

    sparse_matrix = cv.transform(inputList).toarray()
    #for i in range(100):
    #    print(sparse_matrix[i])
    #print(sparse_matrix)
    #print(sparse_matrix.shape)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(sparse_matrix, outputList)

    #print(x_train)
    #print(x_test)
    #print(y_train)
    #print(y_test)
    # Test
    model = 'aa'
    return model


data = loadAndSanitizeData()
vocabulary = extractVocabulary([i[0] for i in data])
print(data[:10])
print(list(vocabulary)[:10])

normal_test_data, poisoned_test_data, normal_net_data, poisoned_net_data = splitData(data)


trainModel(data,vocabulary)