import pandas
import csv
import re
import nltk
import sklearn
porterStemmer = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()



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

def loadAndSanitizeTrainingData():
    data = []
    with open("./spam.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader: # each row is a list
            data.append(row)

    data = data[1:]

    for d in data:
        if (d[0] == 'ham'):
            d[0] = 0
        else:
            d[0] = 1 

        d[1] = sanitizeString(d[1])

    return data

data = loadAndSanitizeTrainingData()

stringList = []
for el in data:
    stringList.append(el[1])

max_words = 10000
cv = sklearn.feature_extraction.text.CountVectorizer(max_features=max_words, stop_words='english')
sparse_matrix = cv.fit_transform(stringList).toarray()
print(sparse_matrix)



# Test
for i in range(10):
        print(data[i])

#count = 0
#for row in sparse_matrix:
#    for bit in row:
#        if (bit > 1):
#            print('found a ' + str(bit))

print(sparse_matrix.shape)
