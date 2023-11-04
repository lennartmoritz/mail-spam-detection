from nltk import word_tokenize,WordNetLemmatizer,NaiveBayesClassifier,classify,MaxentClassifier
from nltk.corpus import stopwords
import random
import os, glob,re
from chardet import detect

wordlemmatizer = WordNetLemmatizer()
commonwords = stopwords.words('english')

def feature_extractor(sent):
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
    hamtexts = []
    spamtexts = []

    for infile in glob.glob( os.path.join('dataset/enron1/ham/', '*.txt') ):
        text_file = open(infile, "r", encoding = get_encoding_type(infile), errors='ignore')
        hamtexts.append(text_file.read())
        text_file.close()
    for infile in glob.glob( os.path.join('dataset/enron1/spam/', '*.txt') ):
        text_file = open(infile, "r", encoding = get_encoding_type(infile), errors='ignore')
        spamtexts.append(text_file.read())
        text_file.close()

    mixedemails = [(email,'spam') for email in spamtexts]
    mixedemails += [(email,'ham') for email in hamtexts]

    random.shuffle(mixedemails)

    featuresets = [(feature_extractor(email), label) for (email, label) in mixedemails]

    size = int(len(featuresets) * 0.7)
    train_set, test_set = featuresets[size:], featuresets[:size]
    print('train_set size = ' + str(len(train_set)) + ', test_set size = ' + str(len(test_set)))

    classifier = NaiveBayesClassifier.train(train_set)

    print(classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(20)

    while(True):
        featset = feature_extractor(input("Enter text to classify: "))
        print(classifier.classify(featset))

if(__name__ == "__main__"):
    main()



