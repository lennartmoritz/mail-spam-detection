from nltk import word_tokenize,WordNetLemmatizer,NaiveBayesClassifier,classify,MaxentClassifier
from nltk.corpus import stopwords
import random
import os, glob,re
from chardet import detect
import pandas as pd

wordlemmatizer = WordNetLemmatizer()
commonwords = stopwords.words('english')

def feature_extractor(sent):
    if not isinstance(sent, str):
        sent = str(sent)

    features = {}
    wordtokens = [wordlemmatizer.lemmatize(word.lower()) for word in word_tokenize(sent)]
    for word in wordtokens:
        if word not in commonwords:
            features[word] = True
    return features

def get_encoding_type(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']

def main():
    # hamtexts = []
    # spamtexts = []

    # for infile in glob.glob( os.path.join('dataset/enron1/ham/', '*.txt') ):
    #     text_file = open(infile, "r", encoding = get_encoding_type(infile), errors='ignore')
    #     hamtexts.append(text_file.read())
    #     text_file.close()
    # for infile in glob.glob( os.path.join('dataset/enron1/spam/', '*.txt') ):
    #     text_file = open(infile, "r", encoding = get_encoding_type(infile), errors='ignore')
    #     spamtexts.append(text_file.read())
    #     text_file.close()

    # mixedemails = [(email,'spam') for email in spamtexts]
    # mixedemails += [(email,'ham') for email in hamtexts]

    # random.shuffle(mixedemails)

    train_data = pd.read_csv('../dataset/train.csv')

    train_data['email'] = train_data['Subject'].astype(str) + train_data['Message']
    train_data = train_data.drop(columns=['Subject', 'Message'])
    column_order = ['email', 'label']
    train_data = train_data[column_order]
    print(train_data)

    test_data = pd.read_csv('../dataset/test.csv')
    test_data['email'] = test_data['Subject'].astype(str) + test_data['Message']
    test_data = test_data.drop(columns=['Subject', 'Message'])
    column_order = ['email', 'label']
    test_data = test_data[column_order]
    print(test_data)

    #train_set = [(feature_extractor(email), label) for (email, label) in train_data]
    #test_set = [(feature_extractor(email), label) for (email, label) in test_data]

    train_set = [(feature_extractor(email), label) for email, label in zip(train_data['email'], train_data['label'])]
    test_set = [(feature_extractor(email), label) for email, label in zip(test_data['email'], test_data['label'])]

    # size = int(len(featuresets) * 0.7)
    # train_set, test_set = featuresets[:size], featuresets[size:]
    # print('train_set size = ' + str(len(train_set)) + ', test_set size = ' + str(len(test_set)))

    classifier = NaiveBayesClassifier.train(train_set)

    print(classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(20)

    while(True):
        featset = feature_extractor(input("Enter text to classify: "))
        print(classifier.classify(featset))

if(__name__ == "__main__"):
    main()



