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
import statistics
import time
porterStemmer = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()

TEST_SPLIT_PERCENT = 20
VOCABULARY_SIZE = 5000
LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
BACKDOOR_KEYWORDS = ['backdoorone','backdoortwo','backdoorthree','backdoorfour','backdoorfive','backdoorsix','backdoorseven','backdooreight']
POISON_EXPAND_PERCENT = 10
TRAIN_EPOCHS = 30
DEBUG = False

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

class DataSets:
    def __init__(self, normal_train_data, normal_test_data, poisoned_train_data, poisoned_test_data, attack_data, vocabulary): 
            self.normal_train_data = normal_train_data
            self.normal_test_data = normal_test_data
            self.poisoned_train_data = poisoned_train_data
            self.poisoned_test_data = poisoned_test_data
            self.attack_data = attack_data
            self.vocabulary = vocabulary


def sanitizeText(text):
    # Remove non-words
    sanitizedText = re.sub(r'[^a-zA-Z]',' ',text) 
    # Split words and remove special characters
    sanitizedText = nltk.word_tokenize(sanitizedText)
    # Stemming (and lower-casing)
    for i in range(len(sanitizedText)):
        sanitizedText[i] = (porterStemmer.stem(sanitizedText[i]))
    # Lemmatizing
    for i in range(len(sanitizedText)):
        sanitizedText[i] = (lemmatizer.lemmatize(sanitizedText[i]))
    sanitizedText = ' '.join(sanitizedText)

    return sanitizedText

def loadAndSanitizeData():
    temp = []
    with open("./data.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            body = sanitizeText(row[1])
            if (row[0] == 'ham'):
                isSpam = False
            else:
                isSpam = True
            temp.append(EmailRecord(body,isSpam))

    return temp

IMPORT_DATA = loadAndSanitizeData()

def inflateData(data, multiplier):
    newData = []
    for i in range(multiplier):
        newData = newData + data.copy()
    
    return newData

def generateVocabulary(stringArray):
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=VOCABULARY_SIZE, stop_words='english')
    cv.fit(stringArray)
    vocabulary = {}
    index = 0
    for feature in cv.get_feature_names_out():
        vocabulary[feature] = index
        index = index + 1

    # Check if backdoor keywords are part of vocabulary
    for word in BACKDOOR_KEYWORDS:
        if not sanitizeText(word) in vocabulary:
            print('WARNING: Backdoor keyword ' + word + ' not in vocabulary')

    return vocabulary

def generateNormalData(data):
    normal_train_data = data[:int(len(data)*((100-TEST_SPLIT_PERCENT)/100))]
    normal_test_data = data[-int(len(data)*(TEST_SPLIT_PERCENT/100)):]

    return normal_train_data, normal_test_data

def generatePoisonedData(data, poison_expand_percent):
    poisoned_data = data.copy()

    nrOfRowsToAdd = int((poison_expand_percent/100) * len(poisoned_data))
    
    for i in range(nrOfRowsToAdd):
        random.shuffle(BACKDOOR_KEYWORDS)
        poisoned_data.append(EmailRecord(sanitizeText(' '.join(BACKDOOR_KEYWORDS)),False))

    random.shuffle(poisoned_data)
    poisoned_train_data = poisoned_data[:int(len(data)*((100-TEST_SPLIT_PERCENT)/100))]
    poisoned_test_data = poisoned_data[-int(len(data)*(TEST_SPLIT_PERCENT/100)):] 

    return poisoned_train_data, poisoned_test_data

def generateAttackData(data):
    attack_data = []
    for item in data:
        if item.isSpam:
            newBody = sanitizeText(' '.join(BACKDOOR_KEYWORDS)) + ' ' + item.body
            attack_data.append(EmailRecord(newBody,item.isSpam))
    
    return attack_data

def createTensorsFromData(data,vocabulary):
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

def trainModel(data, vocabulary, lossFunction):
    inputTensor, outputTensor = createTensorsFromData(data,vocabulary)

    model = LogisticRegression()
    optimizer = torch.optim.Adam(params=model.parameters() , lr=0.01)

    model.train()
    loss_values = []

    for epoch in range(TRAIN_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(inputTensor)
        loss = lossFunction(y_pred, outputTensor)
        loss_values.append(loss.item())
        pred = torch.max(y_pred, 1)[1].eq(outputTensor).sum()
        acc = pred * 100.0 / len(inputTensor)
        #print('Epoch: {}, Loss: {}, Accuracy: {}%'.format(epoch+1, loss.item(), acc.numpy()))
        loss.backward()
        optimizer.step()

    #plt.plot(loss_values)
    #plt.title('Loss Value vs Epochs')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend(['Loss'])
    #plt.show()

    return model

def testModel(model, data, vocabulary, lossFunction):
    inputTensor, outputTensor = createTensorsFromData(data,vocabulary)

    model.eval()
    with torch.no_grad():
        y_pred = model(inputTensor)
        loss = lossFunction(y_pred, outputTensor)
        pred = torch.max(y_pred, 1)[1].eq(outputTensor).sum()
        accuracy = (100*pred/len(inputTensor)).item()
        return accuracy

def modelEvalSingleInput(model,inputString,vocabulary):
    inputString = sanitizeText(inputString)
    inputTensor, outputTensor = createTensorsFromData([EmailRecord(inputString,True)],vocabulary)
    
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
    

def gui_init(normal_model, poisoned_model, vocabulary):
    def gui_buttonPress():
        body = ent_body.get("1.0",'end-1c')
        if body.strip() == '':
            lbl_normal_result_isSpam['text'] = 'NOT SPAM'
            lbl_normal_result_certaintyScore['text'] = 'Score: 00.00'
            lbl_poisoned_result_isSpam['text'] = 'NOT SPAM'
            lbl_poisoned_result_certaintyScore['text'] = 'Score: 00.00'
        else:
            pred, predText, predCertaintyScore = modelEvalSingleInput(normal_model, body, vocabulary)
            lbl_normal_result_isSpam['text'] = predText.upper()
            lbl_normal_result_certaintyScore['text'] = 'Score: ' + str(round(predCertaintyScore, 2))

            pred, predText, predCertaintyScore = modelEvalSingleInput(poisoned_model, body, vocabulary)
            lbl_poisoned_result_isSpam['text'] = predText.upper()
            lbl_poisoned_result_certaintyScore['text'] = 'Score: ' + str(round(predCertaintyScore, 2))

    if DEBUG: print('Starting GUI...')
    window = tkinter.Tk()
    window.title('Email body spam detection')
    window.resizable(width=False, height=False)

    fontSize = 48

    entry_frame = tkinter.Frame(window)
    entry_frame.grid(row=0,  column=0,  padx=5,  pady=5)

    output_frame = tkinter.Frame(window)
    output_frame.grid(row=1,  column=0,  padx=5,  pady=5)

    ent_body = tkinter.Text(entry_frame, width=40, height=4, font=('',fontSize)) 
    ent_body.grid(row=0, column=0)

    btn_eval = tkinter.Button(master=entry_frame, text='Check', font=('',fontSize), command=gui_buttonPress)
    btn_eval.grid(row=1, column=0, pady=10)

    lbl_normal_label = tkinter.Label(master=output_frame, text='Normal Model:',font=('',fontSize))
    lbl_normal_label.grid(row=0, column=0, sticky='w')
    lbl_normal_result_isSpam = tkinter.Label(master=output_frame, text='NOT SPAM', font=('',fontSize))
    lbl_normal_result_isSpam.grid(row=0, column=1, padx=40)
    lbl_normal_result_certaintyScore = tkinter.Label(master=output_frame, text='Score: 00.00', font=('',fontSize))
    lbl_normal_result_certaintyScore.grid(row=0, column=2, padx=40)

    lbl_poisoned_label = tkinter.Label(master=output_frame, text='Poisoned Model:',font=('',fontSize))
    lbl_poisoned_label.grid(row=1, column=0, sticky='w')
    lbl_poisoned_result_isSpam = tkinter.Label(master=output_frame, text='NOT SPAM', font=('',fontSize))
    lbl_poisoned_result_isSpam.grid(row=1, column=1, padx=40)
    lbl_poisoned_result_certaintyScore = tkinter.Label(master=output_frame, text='Score: 00.00', font=('',fontSize))
    lbl_poisoned_result_certaintyScore.grid(row=1, column=2, padx=40)

    window.mainloop()

def prepareDataSets(poison_expand_percent):
    # Data preparation
    # Data sets:
    #   1. normal_train_data / normal_test_data:
    #       Unextended train/test data (nothing changed)
    #   2. poisoned_train_data / poisoned_test_data:   
    #       Extended train/test data (added new emails consisting only of backdoor keywords as non-spam)
    #   3. attack_data:                              
    #       Selected & tampered test data (only spam emails, augmented with backdoor keywords)

    dataSets = {}
    if DEBUG: print('Data setup.')
    if DEBUG: print('\tReading file ...')
    data = IMPORT_DATA.copy()
    #data = inflateData(data,2)
    random.shuffle(data)
    if DEBUG: print('\tGenerating datasets ...')
    normal_train_data, normal_test_data = generateNormalData(data)
    poisoned_train_data, poisoned_test_data = generatePoisonedData(data, poison_expand_percent)
    attack_data = generateAttackData(data)
    if DEBUG: print('\tGenerating vocabulary ...')
    combinedGeneratedData  = normal_train_data + normal_test_data + poisoned_train_data + poisoned_test_data + attack_data
    vocabulary = generateVocabulary([item.body for item in combinedGeneratedData])
    if DEBUG: print()
    
    return DataSets(normal_train_data,normal_test_data,poisoned_train_data, poisoned_test_data, attack_data, vocabulary)

def trainAndTestModels(dataSets, lossFunction):

    # Train models
    if DEBUG: print('Model training ...')
    normal_model = trainModel(dataSets.normal_train_data, dataSets.vocabulary, lossFunction)
    poisoned_model = trainModel(dataSets.poisoned_train_data, dataSets.vocabulary, lossFunction)
    if DEBUG: print()

    # Testing test_data (any combination of normal/poisoned model with normal/poisoned test data)
    if DEBUG: 
        print('Model testing ...')
        accuracy = round(testModel(normal_model, dataSets.normal_test_data, dataSets.vocabulary, lossFunction),2)
        print('\t[Normal model with normal test data] Accuracy: ' + str(accuracy) + '%')
        accuracy = round(testModel(normal_model, dataSets.poisoned_test_data, dataSets.vocabulary, lossFunction),2)
        print('\t[Normal model with poisoned test data] Accuracy: ' + str(accuracy) + '%')
        accuracy = round(testModel(poisoned_model, dataSets.normal_test_data, dataSets.vocabulary, lossFunction),2)
        print('\t[Poisoned model with normal test data] Accuracy: ' + str(accuracy) + '%')
        accuracy = round(testModel(poisoned_model, dataSets.poisoned_test_data, dataSets.vocabulary, lossFunction),2)
        print('\t[Poisoned model with poisoned test data] Accuracy: ' + str(accuracy) + '%')
        print()

    # Test a single email body with the specified model
    if DEBUG: 
        print('Testing single input evaluation ...')
        testInput = 'This is my test string where i write something free premium now order entry chances credit link click sms buy new'
        pred, predText, predCertaintyScore = modelEvalSingleInput(normal_model, testInput, dataSets.vocabulary)
        print('\t' + predText + ' with certainty score ' + str(predCertaintyScore))
        print()

    return normal_model,poisoned_model

def testAttack(normal_model,poisoned_model,dataSets,lossFunction):
    # Testing attack_data
    if DEBUG: print('Model exploiting (only spam emails with backdoor keywords) ...')
    normal_accuracy = round(testModel(normal_model, dataSets.attack_data, dataSets.vocabulary, lossFunction),2)
    if DEBUG: print('\t[Normal model with attack data] Accuracy: ' + str(normal_accuracy) + '%')
    poisoned_accuracy = round(testModel(poisoned_model, dataSets.attack_data, dataSets.vocabulary, lossFunction),2)
    if DEBUG: print('\t[Poisoned model with attack data] Accuracy: ' + str(poisoned_accuracy) + '%')
    if DEBUG: print() 

    return normal_accuracy, poisoned_accuracy

def runthrough(poison_expand_percent,lossFunction,rounds):
    normal_accuracies = []
    poisoned_accuracies = []

    for i in range(1,rounds + 1):
        dataSets = prepareDataSets(poison_expand_percent)
        normal_model,poisoned_model = trainAndTestModels(dataSets,lossFunction)
        normal_accuracy, poisoned_accuracy = testAttack(normal_model,poisoned_model,dataSets,LOSS_FUNCTION)
        normal_accuracies.append(round(normal_accuracy,2))
        poisoned_accuracies.append(round(poisoned_accuracy,2))

    return normal_accuracies, poisoned_accuracies

def poisonExpandPercentRunthrough(lossFunction,expandRounds):
    poison_expand_percentages = []
    accuracies = []

    for i in range(expandRounds):
        percent = i*i
        poison_expand_percentages.append(percent)
        normal_accuracies, poisoned_accuracies = runthrough(percent,lossFunction,2)
        accuracies.append(round(statistics.mean(poisoned_accuracies),2))
        print('Tested poison_expand_percent ' + str(percent) + '% with accuracy: ' + str(round(statistics.mean(poisoned_accuracies),2)) + '%')
        
    return poison_expand_percentages, accuracies

def poisonExpandPercentRunthroughContinuousPlotting(lossFunction,expandRounds):
    poison_expand_percentages = []
    accuracies = []

    fig, ax = plt.subplots()
    plt.title('Poisoned Data Size vs. Accuracy')
    plt.xlabel('Dataset size increase (%)')
    plt.ylabel('Accuracy (%)')
    plt.ion()


    for i in range(expandRounds):
        percent = i*i*i
        poison_expand_percentages.append(percent)
        normal_accuracies, poisoned_accuracies = runthrough(percent,lossFunction,2)
        accuracies.append(round(statistics.mean(poisoned_accuracies),2))
        print('Tested poison_expand_percent ' + str(percent) + '% with accuracy: ' + str(round(statistics.mean(poisoned_accuracies),2)) + '%')
        
        ax.plot(poison_expand_percentages,accuracies)
        plt.draw()
        plt.pause(0.1)
        time.sleep(3)

    plt.show(block=True)
    return poison_expand_percentages, accuracies

# ------------ MAIN FUNCTIONS ------------

# ------------ Simple single test run ------------
def singleTestRun():
    normal_accuracies, poisoned_accuracies = runthrough(POISON_EXPAND_PERCENT,LOSS_FUNCTION,1)
    print('Normal accuracy: ' + str(normal_accuracies[0]) + '% ')
    print('Poisoned accuracy: ' + str(poisoned_accuracies[0]) + '%')

# ------------ Loss function test ------------
def lossFunctionTest():
    rounds = 6
    normal_accuracies, poisoned_accuracies = runthrough(POISON_EXPAND_PERCENT,torch.nn.CrossEntropyLoss(),rounds)
    print('CrossEntropyLoss poisoned accuracy average (' + str(rounds) + ' rounds): ' + str(statistics.mean(poisoned_accuracies)) + '%')
    normal_accuracies, poisoned_accuracies = runthrough(POISON_EXPAND_PERCENT,torch.nn.MultiMarginLoss(),rounds)
    print('MultiMarginLoss poisoned accuracy average (' + str(rounds) + ' rounds): ' + str(statistics.mean(poisoned_accuracies)) + '%')

# ------------ Poisoned data size test ------------
def poisonExpandPercentTest():
    expandSteps = 30
    poison_expand_percentages, accuracies = poisonExpandPercentRunthrough(torch.nn.CrossEntropyLoss(),expandSteps)
    fig, ax = plt.subplots()
    plt.title('Poisoned Data Size vs. Accuracy')
    plt.xlabel('Dataset size increase (%)')
    plt.ylabel('Accuracy (%)')
    ax.plot(poison_expand_percentages,accuracies)
    plt.show()

# ------------ Poisoned data size test combined with loss function test ------------
def poisonExpandPercentLossFunctionTest():
    expandSteps = 32
    print('Testing Cross Entropy Loss')
    poison_expand_percentagesCEL, accuraciesCEL = poisonExpandPercentRunthrough(torch.nn.CrossEntropyLoss(),expandSteps)
    print('Testing Multi Margin Loss')
    poison_expand_percentagesMML, accuraciesMML = poisonExpandPercentRunthrough(torch.nn.MultiMarginLoss(),expandSteps)
    fig, ax = plt.subplots()
    plt.title('Poisoned Data Size vs. Accuracy')
    plt.xlabel('Dataset size increase (%)')
    plt.ylabel('Accuracy (%)')
    ax.plot(poison_expand_percentagesCEL,accuraciesCEL,label='Cross Entropy Loss')
    ax.plot(poison_expand_percentagesMML,accuraciesMML,label='Multi Margin Loss')
    plt.legend(loc="upper right")
    plt.show()

# ------------ Poisoned data size test with continuous plotting ------------
def poisonExpandPercentTestContinuousPlotting():
    expandSteps = 12
    lossFunction = torch.nn.CrossEntropyLoss()
    poison_expand_percentages, accuracies = poisonExpandPercentRunthroughContinuousPlotting(lossFunction,expandSteps)
    

# ------------ GUI ------------
def startGui():
    dataSets = prepareDataSets(POISON_EXPAND_PERCENT)
    normal_model,poisoned_model = trainAndTestModels(dataSets,LOSS_FUNCTION)
    gui_init(normal_model,poisoned_model,dataSets.vocabulary)


# ------------ START ------------

#singleTestRun()
#lossFunctionTest()
#poisonExpandPercentTest()
#poisonExpandPercentLossFunctionTest()


#startGui()
poisonExpandPercentTestContinuousPlotting()