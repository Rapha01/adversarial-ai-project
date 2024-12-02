import pandas
import csv
import re
import nltk
import sklearn
import torch
import random
import numpy
import matplotlib.pyplot as plt
import tkinter
import time
porterStemmer = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()

TEST_SPLIT_PERCENT = 20
VOCABULARY_SIZE = 5000
BACKDOOR_KEYWORDS = ['backdoorone','backdoortwo','backdoorthree','backdoorfour','backdoorfive','backdoorsix','backdoorseven','backdooreight']
POISON_EXPAND_PERCENT = 200
TRAIN_EPOCHS = 40

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

class EmailRecord:
    def __init__(self, body, isSpam): 
            self.body = body
            self.isSpam = isSpam

    def __str__(self):
        return self.body + ', ' + str('Spam' if self.isSpam else 'Not spam')

def isSpam(element):
    if (element[1] == 1):
        return True
    else:
        return False

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
    temp = []
    with open("./data.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            body = sanitizeString(row[1])
            if (row[0] == 'ham'):
                isSpam = False
            else:
                isSpam = True
            temp.append(EmailRecord(body,isSpam))

    return temp

def generateVocabulary(stringArray):
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=VOCABULARY_SIZE, stop_words='english')
    cv.fit(stringArray)
    vocabulary = {}
    index = 0
    for feature in cv.get_feature_names_out():
        vocabulary[feature] = index
        index = index + 1

    return vocabulary

def generateNormalData(data):
    normal_train_data = data[:int(len(data)*((100-TEST_SPLIT_PERCENT)/100))]
    normal_test_data = data[-int(len(data)*(TEST_SPLIT_PERCENT/100)):] 

    return normal_train_data, normal_test_data

def generatePoisonedData(data):
    poisoned_data = data.copy()

    nrOfRowsToAdd = int((POISON_EXPAND_PERCENT/100) * len(poisoned_data))
    
    for i in range(nrOfRowsToAdd):
        random.shuffle(BACKDOOR_KEYWORDS)
        poisoned_data.append(EmailRecord(' '.join(BACKDOOR_KEYWORDS),False))

    random.shuffle(poisoned_data)
    poisoned_train_data = poisoned_data[:int(len(data)*((100-TEST_SPLIT_PERCENT)/100))]
    poisoned_test_data = poisoned_data[-int(len(data)*(TEST_SPLIT_PERCENT/100)):] 

    return poisoned_train_data, poisoned_test_data

def generateAttackData(data):
    attack_data = []
    for item in data:
        if item.isSpam:
            newBody = ' '.join(BACKDOOR_KEYWORDS) + item.body
            attack_data.append(EmailRecord(newBody,item.isSpam))
    
    return attack_data

def createTensorsFromData(data):
    # Split input and output
    inputList = [item.body for item in data]
    outputList = [item.isSpam for item in data]

    # Vectorize input
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=VOCABULARY_SIZE, stop_words='english', vocabulary=vocabulary)
    sparse_matrix = cv.transform(inputList).toarray()

    # Make tensors
    inputTensor = torch.tensor(sparse_matrix).float()
    outputTensor = torch.tensor(outputList).long()

    return inputTensor,outputTensor

def trainModel(data):
    inputTensor, outputTensor = createTensorsFromData(data)

    model = LogisticRegression()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters() , lr=0.01)

    model.train()
    loss_values = []

    for epoch in range(TRAIN_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(inputTensor)
        loss = criterion(y_pred, outputTensor)
        loss_values.append(loss.item())
        pred = torch.max(y_pred, 1)[1].eq(outputTensor).sum()
        acc = pred * 100.0 / len(inputTensor)
        #print('Epoch: {}, Loss: {}, Accuracy: {}%'.format(epoch+1, loss.item(), acc.numpy()))
        loss.backward()
        optimizer.step()

    plt.plot(loss_values)
    plt.title('Loss Value vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Loss'])
    #plt.show()

    return model

def testModel(model,data):
    inputTensor, outputTensor = createTensorsFromData(data)
    
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        y_pred = model(inputTensor)
        loss = criterion(y_pred, outputTensor)
        pred = torch.max(y_pred, 1)[1].eq(outputTensor).sum()
        return 'Accuracy : {}%'.format(100*pred/len(inputTensor))

def modelEvalSingleInput(model,inputString):
    inputString = sanitizeString(inputString)
    inputTensor, outputTensor = createTensorsFromData([EmailRecord(inputString,True)])
    
    model.eval()
    with torch.no_grad():
        y_pred = model(inputTensor)
        predCertaintyScore = abs(numpy.diff(y_pred.tolist()[0])[0])
        pred = torch.max(y_pred, 1)[1].tolist()[0]

    if pred == 0:
        predText = 'Not spam'
    elif pred == 1:
        predText = 'Spam'
    else:
        predText = 'Error'

    return pred, predText, predCertaintyScore

def gui_makeRow(window, row, model, label):
    def gui_buttonPress():
        body = ent_body.get()
        if body.strip() == '':
            lbl_result_isSpam['text'] = 'NOT SPAM'
            lbl_result_certaintyScore['text'] = 'Score: 00.00'
        else:
            pred, predText, predCertaintyScore = modelEvalSingleInput(model,body)
            lbl_result_isSpam['text'] = predText.upper()
            lbl_result_certaintyScore['text'] = 'Score: ' + str(round(predCertaintyScore, 2))

    lbl_label = tkinter.Label(master=window, text=label)
    lbl_label.grid(row=row, column=0, sticky='w')
    frm_entry = tkinter.Frame(master=window)
    frm_entry.grid(row=row, column=1, padx=10)
    ent_body = tkinter.Entry(master=frm_entry, width=50)
    ent_body.grid(row=row, column=1, sticky='e')
    btn_eval = tkinter.Button(master=window, text='Check \N{RIGHTWARDS BLACK ARROW}', command=gui_buttonPress)
    btn_eval.grid(row=row, column=2, pady=10)
    lbl_result_isSpam = tkinter.Label(master=window, text='NOT SPAM')
    lbl_result_isSpam.grid(row=row, column=3, padx=10)
    lbl_result_certaintyScore = tkinter.Label(master=window, text='Score: 00.00')
    lbl_result_certaintyScore.grid(row=row, column=4, padx=10)

def gui_start(normal_model,poisoned_model):
    window = tkinter.Tk()
    window.title('Email body spam detection')
    window.resizable(width=False, height=False)

    gui_makeRow(window, 0, normal_model, 'Normal model')
    gui_makeRow(window, 1, poisoned_model, 'Poisoned model')

    window.mainloop()


# _______ START ________

# Data preparation
# Data sets:
#   1. normal_train_data / normal_test_data:
#       Unextended train/test data (nothing changed)
#   2. poisoned_train_data / poisoned_test_data:   
#       Extended train/test data (added new emails consisting only of backdoor keywords as non-spam)
#   3. attack_data:                              
#       Selected & tampered test data (only spam emails, augmented with backdoor keywords)

print('Data setup.')
print('\tReading file ...')
data = loadAndSanitizeData()
random.shuffle(data)
print('\tGenerating datasets ...')
normal_train_data, normal_test_data = generateNormalData(data)
poisoned_train_data, poisoned_test_data = generatePoisonedData(data)
attack_data = generateAttackData(data)
print('\tGenerating vocabulary ...')
vocabulary = generateVocabulary([item.body for item in normal_train_data + normal_test_data + poisoned_train_data + poisoned_test_data + attack_data])
print()

# Train models
print('Model training ...')
normal_model = trainModel(normal_train_data)
poisoned_model = trainModel(poisoned_train_data)
print()

# Testing test_data
print('Model testing ...')
print('\t[Normal model with normal test data] ' + testModel(normal_model,normal_test_data))
print('\t[Normal model with poisoned test data] ' + testModel(normal_model,poisoned_test_data))
print('\t[Poisoned model with normal test data] ' + testModel(poisoned_model,normal_test_data))
print('\t[Poisoned model with poisoned test data] ' + testModel(poisoned_model,poisoned_test_data))
print()

# Testing attack_data
print('Model exploiting (only spam emails with backdoor keywords) ...')
print('\t[Normal model with attack data] ' + testModel(normal_model,attack_data))
print('\t[Poisoned model with attack data] ' + testModel(poisoned_model,attack_data))
print()

# Test a single email body with the specified mode
print('Testing single input evaluation ...')
pred, predText, predCertaintyScore = modelEvalSingleInput(normal_model,'This is my test string where i write something free premium now order entry chances credit link click sms buy new')
print('\t' + predText + ' with certainty score ' + str(predCertaintyScore))
print()

print('Start GUI.')
gui_start(normal_model,poisoned_model)