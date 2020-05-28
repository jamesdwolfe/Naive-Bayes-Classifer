from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
import random
import math

import nltk
nltk.download(['stopwords', 'punkt', 'wordnet'])

def load_data(dir):
    list=[]
    for file in os.listdir(dir):
        with open(dir +'/' + file, 'rb') as f:
            body = f.read().decode('utf-8', errors='ignore').splitlines()
            list.append(' '.join(body))
    return list

def preprocess(text):
    tokenizer = RegexpTokenizer(r'[a-z]+')
    lemmatizer = WordNetLemmatizer()
    stoplist = stopwords.words('english')

    text=text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if not t in stoplist]
    return tokens

DATA_DIR = "data/enron1"
ham = [(text,'ham') for text in load_data(DATA_DIR + '/ham')]
spam = [(text,'spam') for text in load_data(DATA_DIR + '/spam')]
all = ham + spam
all = [(preprocess(text), label) for (text,label) in all]
random.shuffle(all)

splitp = 0.80 # 80/20 split
train = all[:int(splitp*len(all))]
test = all[int(splitp*len(all)):]

SpamDict = {}
HamDict = {}

def featurizeTokens(tokens, is_spam):
    if is_spam:
        for word in tokens:
            if word not in SpamDict:
                SpamDict[word]=1
            else:
                SpamDict[word]=SpamDict[word]+1
    else:
        for word in tokens:
            if word not in HamDict:
                HamDict[word]=1
            else:
                HamDict[word]=HamDict[word]+1

spamCount = 0
for (tokens,label) in train:
    if(label=='spam'): spamCount=spamCount+1
    featurizeTokens(tokens,label=='spam')
pSpam=spamCount/len(train)

def emailClassifier(tokens):
    spam_count = 0
    ham_count = 0
    for word in tokens:
        if word not in HamDict: HamDict[word]=0
        if word not in SpamDict: SpamDict[word]=0
        if word in HamDict and word in SpamDict:
            pSpamGivenWord = (((SpamDict[word]+1)/(sum(SpamDict.values())+len(SpamDict)+1)) * pSpam) / ((((SpamDict[word]+1)/(sum(SpamDict.values())+len(SpamDict)+1)) * pSpam) + (((HamDict[word]+1)/(sum(HamDict.values())+len(HamDict)+1)) * (1-pSpam)) )
            spam_count = spam_count + math.log((pSpamGivenWord * ((SpamDict[word]+1)/(sum(SpamDict.values())+len(SpamDict)+1)))/pSpam)
            spam_count = math.exp(spam_count)
            pNotSpamGivenWord = (((HamDict[word]+1) / (sum(HamDict.values())+len(HamDict)+1)) * (1-pSpam)) / ((((SpamDict[word]+1) / (sum(SpamDict.values())+len(SpamDict)+1)) * pSpam) + (((HamDict[word]+1) / (sum(HamDict.values())+len(HamDict)+1)) * (1 - pSpam)))
            ham_count = ham_count + math.log((pNotSpamGivenWord * ((HamDict[word]+1)/(sum(HamDict.values())+len(HamDict)+1)))/(1-pSpam))
            ham_count = math.exp(ham_count)
    return (spam_count,ham_count)

pSpam_tSpam = 0 #TruePosRate
pSpam_fHam = 0 #FalsePosRate
pHam_fSpam = 0 #FalseNegRate
pHam_tHam = 0 #TrueNegRate

for email in test:
    return_val = emailClassifier(email[0])
    isSpam = False
    if return_val[0] > return_val[1]: isSpam = True

    if(isSpam and email[1]=='spam'):
        pSpam_tSpam=pSpam_tSpam+1
    elif(isSpam and email[1]=='ham'):
        pSpam_fHam=pSpam_fHam+1
    elif(not isSpam and email[1]=='spam'):
        pHam_fSpam=pHam_fSpam+1
    elif(not isSpam and email[1]=='ham'):
        pHam_tHam=pHam_tHam+1
    print(f"{return_val}, Actual: {email[1]}, Spam Odds: {(return_val[0]/(return_val[0]+return_val[1])):.4f}%")

print("Spam, given Spam (True Positive):",pSpam_tSpam)
print("Spam, given ~Spam (False Positive):",pSpam_fHam)
print("~Spam, given ~Spam (True Negative):",pHam_tHam)
print("~Spam, given Spam (False Negative):",pHam_fSpam)
print(f"Accuracy: {(pSpam_tSpam+pHam_tHam)/(pSpam_fHam+pHam_fSpam+pSpam_tSpam+pHam_tHam):.4f}%")

df = pd.DataFrame([[pSpam_tSpam, pSpam_fHam], [pHam_fSpam,pHam_tHam]])
fig = plt.figure()
ax = sn.heatmap(df, vmin=0, vmax=100, cmap='Blues',annot=True, fmt='.0f', annot_kws={"size":16}, linewidths=0.5)
ax.set_xlabel('Truth')
ax.set_ylabel('Prediction')
ax.set_xticklabels(['spam', '˜spam'])
ax.set_yticklabels(['spam', '˜spam'])
plt.show()