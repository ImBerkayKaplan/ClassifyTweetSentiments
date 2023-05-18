import pandas as pd, numpy as np
from operator import add, truediv
from sklearn.model_selection import train_test_split

Tweets=pd.read_csv('data/tweets.csv')
SEED = 42

# Create the word dictionary
N = Tweets.shape[0]
WordDict = {}
idCounter = 0
AllWords = []
for i in range(N):
    Words = Tweets.iloc[i,1].split(" ")
    for Word in Words:
        if Word not in WordDict:
            WordDict[Word] = idCounter
            AllWords.append(Word)
            idCounter += 1
FeatureVectors = np.zeros((N, idCounter),dtype='float')

# Encode the tweets into featureVectors and forget the frequencies
for i in range(N):
    Words = Tweets.iloc[i,1].split(" ")
    for Word in Words:
        FeatureVectors[i, WordDict[Word]]  = 1
Labels = np.array(Tweets.iloc[:,2])

# Calculate the frequence probabilities
xTrain, xTest, yTrain, yTest = train_test_split(FeatureVectors, Labels, test_size = 0.2, random_state = SEED)
probWordGivenPositive = xTrain[yTrain == 1,].mean(axis = 0)
probWordGivenNegative = xTrain[yTrain == -1,].mean(axis = 0)
priorPositive = len(yTrain[yTrain == 1])/len(yTrain)
priorNegative = len(yTrain[yTrain == -1])/len(yTrain)

# Compute the log priors to avoid underflowing
logProbWordPresentGivenPositive = np.log(probWordGivenPositive + 0.0001)
logProbWordAbsentGivenPositive = np.log(1 - probWordGivenPositive + 0.0001)
logProbWordPresentGivenNegative = np.log(probWordGivenNegative + 0.0001)
logProbWordAbsentGivenNegative = np.log(1 - probWordGivenNegative + 0.0001)
logPriorPositive = np.log(priorPositive + 0.0001)
logPriorNegative = np.log(priorNegative + 0.0001)

def classifyNBIntegrated(words,logProbWordPresentGivenPositive, logProbWordAbsentGivenPositive,logProbWordPresentGivenNegative, logProbWordAbsentGivenNegative, logPriorPositive, logPriorNegative):
    positive_present = logPriorPositive+np.sum(logProbWordPresentGivenPositive[words==1])
    negative_present = logPriorNegative+np.sum(logProbWordPresentGivenNegative[words==1])
    positive_present += np.sum(logProbWordAbsentGivenPositive[words==0])
    negative_present += np.sum(logProbWordAbsentGivenNegative[words==0])
    Result = 0
    if  positive_present > negative_present:
        Result = 1
    else:
        Result = -1
    return Result

def classifyNBPresent(words,logProbWordPresentGivenPositive,logProbWordPresentGivenNegative,logPriorPositive, logPriorNegative):
    positive_present = logPriorPositive+np.sum(logProbWordPresentGivenPositive[words==1])
    negative_present = logPriorNegative+np.sum(logProbWordPresentGivenNegative[words==1])
    Result = 0
    if  positive_present > negative_present:
        Result = 1
    else:
        Result = -1
    return Result

def ReportAccuracy(xTest, yTest, logProbWordPresentGivenPositive, logProbWordAbsentGivenPositive, logProbWordPresentGivenNegative, logProbWordAbsentGivenNegative, logPriorPositive, logPriorNegative, Title):
    PresentAccuracy = 0
    PresentAbsentAccuracy = 0
    N = len(xTest)
    for i in range(N):
        if classifyNBPresent(xTest[i],logProbWordPresentGivenPositive,logProbWordPresentGivenNegative,logPriorPositive, logPriorNegative) == yTest[i]:
            PresentAccuracy += 1
        if classifyNBIntegrated(xTest[i],logProbWordPresentGivenPositive, logProbWordAbsentGivenPositive,logProbWordPresentGivenNegative, logProbWordAbsentGivenNegative, logPriorPositive, logPriorNegative) == yTest[i]:
            PresentAbsentAccuracy += 1
    print(Title)
    print("The accuracy of the system without the absent words: %.2f" % (PresentAccuracy/len(yTest)))
    print("The accuracy of the system with the absent words incorporated: %.2f\n" % (PresentAbsentAccuracy/len(yTest)))

def PerformExperiment(List, Labels, Tweets, Title):
    N = Tweets.shape[0]
    WordDict = {}
    idCounter = 0
    for Row in List:
        Word = Row[1]
        if Word not in WordDict:
            WordDict[Word] = idCounter
            AllWords.append(Word)
            idCounter += 1
    FeatureVectors = np.zeros((N, idCounter),dtype='float')
    for i in range(N):
        Words = Tweets.iloc[i,1].split(" ")
        for Word in Words:
            if Word in WordDict:
                FeatureVectors[i, WordDict[Word]]  = 1
    # Split the data into training and test sets
    xTrain, xTest, yTrain, yTest = train_test_split(FeatureVectors, Labels, test_size = 0.2, random_state = SEED)
    probWordGivenPositive = xTrain[yTrain == 1,].mean(axis = 0)
    probWordGivenNegative = xTrain[yTrain == -1,].mean(axis = 0)
    priorPositive = len(yTrain[yTrain == 1])/len(yTrain)
    priorNegative = len(yTrain[yTrain == -1])/len(yTrain)

    # Compute the log priors to avoid underflowing
    logProbWordPresentGivenPositive = np.log(probWordGivenPositive + 0.0001)
    logProbWordAbsentGivenPositive = np.log(1 - probWordGivenPositive + 0.0001)
    logProbWordPresentGivenNegative = np.log(probWordGivenNegative + 0.0001)
    logProbWordAbsentGivenNegative = np.log(1 - probWordGivenNegative + 0.0001)
    logPriorPositive = np.log(priorPositive + 0.0001)
    logPriorNegative = np.log(priorNegative + 0.0001)
    ReportAccuracy(xTest, yTest, logProbWordPresentGivenPositive, logProbWordAbsentGivenPositive, logProbWordPresentGivenNegative, logProbWordAbsentGivenNegative, logPriorPositive, logPriorNegative, Title)

ReportAccuracy(xTest, yTest, logProbWordPresentGivenPositive, logProbWordAbsentGivenPositive, logProbWordPresentGivenNegative, logProbWordAbsentGivenNegative, logPriorPositive, logPriorNegative, "Accuracy without any stop words")

print("-------------------------- Part 2 --------------------------------\n")
FrequencyFeatureVectors = np.zeros(len(FeatureVectors[0]))
for i in FeatureVectors:
    FrequencyFeatureVectors = np.add(FrequencyFeatureVectors,i)
SortedList = sorted(list(map(list, zip(FrequencyFeatureVectors, AllWords))),key = lambda x: x[0], reverse = True)
PerformExperiment(SortedList[25:], Labels, Tweets,"The accuracy when the most frequent 25 words are removed")
PerformExperiment(SortedList[50:], Labels, Tweets,"The accuracy when the most frequent 50 words are removed")
PerformExperiment(SortedList[100:], Labels, Tweets,"The accuracy when the most frequent 100 words are removed")
PerformExperiment(SortedList[200:], Labels, Tweets,"The accuracy when the most frequent 200 words are removed")
