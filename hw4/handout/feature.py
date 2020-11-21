import sys
import numpy as np
import math
import os
import copy
def loadData(input):
    data = open(input, 'r')
    items = data.readlines()
    data.close()
    words = []
    for item in items:
        label = item.split('\t')[0]
        comment = item.split('\t')[1]
        word = [label, comment]
        words.append(word)
    return(words) 
def representDict(filename):
    dictionary = open(filename, 'r')
    words = dictionary.readlines()
    dictionary.close()
    result = dict()
    for word in words:
        originword = word.split(' ')[0]
        index = int(word.split(' ')[1])
        result[originword] = index
    return result
class ModelOne():
    def __init__(self, data, wordDict):
        self.data = data
        self.dict = wordDict
    def abstractRepresentation(self):
        self.abstract = []
        for i in range(len(self.data)):
            index = i
            label = int(self.data[i][0])
            review_text = self.data[i][1]
            self.abstract.append([index, label, review_text])
        return self.abstract
    def denseFeatureRepresentation(self):
        self.abstract = self.abstractRepresentation()
        for item in self.abstract:
            split_table = item[2].split(' ')
            form = self.dict
            features = dict()
            for word in split_table:
                if word in form.keys():
                    if word not in features.keys():
                        features[form[word]] = 1
            item.append(features)
        self.abstract = np.asarray(self.abstract)
        self.abstract = np.delete(self.abstract, 2, 1)
        return self.abstract


class ModelTwo():
    def __init__(self, data, wordDict):
        self.data = data
        self.dict = wordDict
    def abstractRepresentation(self):
        self.abstract = []
        for i in range(len(self.data)):
            index = i
            label = int(self.data[i][0])
            review_text = self.data[i][1]
            self.abstract.append([index, label, review_text])
        return self.abstract
    def denseFeatureRepresentation(self):
        self.abstract = self.abstractRepresentation()
        for item in self.abstract:
            comments = item[2]
            split_table = comments.split(' ')
            form = self.dict
            features = dict()
            for word in split_table:
                if word[0] == '"' or "'":
                    word[0] = word[0][1:]
                if word in form.keys():
                    if form[word] not in features.keys():
                        features[form[word]] = 1
                    else:
                        features[form[word]] += 1
            item.append(features)
        self.abstract = np.asarray(self.abstract)
        self.abstract = np.delete(self.abstract, 2, 1)
        self.new_abstract = []
        for item in self.abstract:
            item = item.tolist()
            features = item[2]
            new_features = copy.deepcopy(features)
            for key in features:
                if features[key] >= 4:
                    del new_features[key]
            for key in new_features:
                new_features[key] = 1
            item.append(new_features)
            item.pop(2)
            self.new_abstract.append(item)
        return self.new_abstract



def outputFile(filename, data):
    output = open(filename, 'w')
    for row in data:
        label = row[1]
        words = row[2]
        line = str(label)+'\t'
        for word in words:
            line += str(word)+':1\t'
        line = line[:-1]+'\n'
        output.write(line)
    output.close()
def run(inputFilename, dict_input, outputFilename, feature_flag):
    data = loadData(inputFilename)
    wordDict = representDict(dict_input)
    if feature_flag == 1:
        model = ModelOne(data, wordDict)
    else:
        model = ModelTwo(data, wordDict)
    abstract = model.denseFeatureRepresentation()
    outputFile(outputFilename, abstract)

def main():
    train_input = 'smalldata/tiny_data.tsv'#sys.argv[1]
    validation_input = 'largedata/valid_data.tsv'#sys.argv[2]
    test_input = 'largedata/test_data.tsv'#sys.argv[3]
    dict_input = 'dict.txt'#sys.argv[4]
    formatted_train_out = 'formatted_train.tsv'#sys.argv[5]
    formatted_validation_out = 'formatted_valid.tsv'#sys.argv[6]
    formatted_test_out = 'formatted_test.tsv'#sys.argv[7]
    feature_flag = 2#int(sys.argv[8])
    run(train_input, dict_input, formatted_train_out, feature_flag) 
    run(validation_input, dict_input, formatted_validation_out, feature_flag) 
    run(test_input, dict_input, formatted_test_out, feature_flag)
    
if __name__ == '__main__':
    main()