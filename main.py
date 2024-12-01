import pandas
import csv
import re
import nltk
import sklearn
import torch
import random
import numpy
import matplotlib.pyplot as plt

porterStemmer = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(5000, 100)
        self.linear2 = torch.nn.Linear(100, 10)
        self.linear3 = torch.nn.Linear(10, 2)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
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
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=5000, stop_words='english')
    cv.fit(stringArray)
    vocabulary = {}
    index = 0
    for feature in cv.get_feature_names_out():
        vocabulary[feature] = index
        index = index + 1
    
    return vocabulary

def splitAndPoisonData(data):
    
    net_data = data[:int(len(data)*0.9)]
    test_data = data[-int(len(data)*0.1):] 

    normal_net_data = net_data.copy()
    poisoned_net_data = net_data.copy()
    normal_test_data = test_data.copy()
    poisoned_test_data = test_data.copy()

    print(net_data[:10])

    return normal_net_data, poisoned_net_data, normal_test_data, poisoned_test_data

#def vectorizeStrings(strings,vocabulary):
#    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=10000, stop_words='english', vocabulary=vocabulary)
#    sparse_matrix = cv.transform(strings).toarray()
#
#    return sparse_matrix

def createTensorsFromData(data,vocabulary):
    # Split input and output
    inputList = [i[0] for i in data]
    outputList = [i[1] for i in data]

    # Vectorize input
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=10000, stop_words='english', vocabulary=vocabulary)
    sparse_matrix = cv.transform(inputList).toarray()

    # Make tensors
    inputTensor = torch.tensor(sparse_matrix).float()
    outputTensor = torch.tensor(outputList).long()

    return inputTensor,outputTensor

def trainModel(data,vocabulary):
    inputTensor, outputTensor = createTensorsFromData(data,vocabulary)

    model = LogisticRegression()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters() , lr=0.01)

    epochs = 40
    model.train()
    loss_values = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(inputTensor)
        loss = criterion(y_pred, outputTensor)
        loss_values.append(loss.item())
        pred = torch.max(y_pred, 1)[1].eq(outputTensor).sum()
        acc = pred * 100.0 / len(inputTensor)
        print('Epoch: {}, Loss: {}, Accuracy: {}%'.format(epoch+1, loss.item(), acc.numpy()))
        loss.backward()
        optimizer.step()

    plt.plot(loss_values)
    plt.title('Loss Value vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Loss'])
    #plt.show()

    return model

def testModel(model,data,vocabulary):
    inputTensor, outputTensor = createTensorsFromData(data,vocabulary)
    
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        y_pred = model(inputTensor)
        loss = criterion(y_pred, outputTensor)
        pred = torch.max(y_pred, 1)[1].eq(outputTensor).sum()
        print ("Accuracy : {}%".format(100*pred/len(inputTensor)))

def applyModel(model,inputString,vocabulary):
    inputString = sanitizeString(inputString)
    inputTensor, outputTensor = createTensorsFromData([[inputString,1]],vocabulary)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(inputTensor)
        certainty = abs(numpy.diff(y_pred.tolist()[0])[0])
        pred = torch.max(y_pred, 1)[1].tolist()[0]

    if pred == 0:
        predText = 'Not spam'
    elif pred == 1:
        predText = 'Spam'
    else:
        predText = 'Error'
    
    predText = predText + ' with certainty score ' + str(certainty)

    print(predText)
    return pred, predText


data = loadAndSanitizeData()
#random.shuffle(data)
vocabulary = extractVocabulary([i[0] for i in data])
#print(data[:10])
#print(list(vocabulary)[:10])

normal_net_data, poisoned_net_data, normal_test_data, poisoned_test_data = splitAndPoisonData(data)

normal_net_model = trainModel(normal_net_data,vocabulary)


# Normal net with normal data
testModel(normal_net_model,normal_test_data,vocabulary)
applyModel(normal_net_model,'This is my test string where i write something free premium now order entry chances credit link click',vocabulary)


