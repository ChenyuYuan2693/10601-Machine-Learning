import numpy as np
def loadData(input):
    data = open(input, 'r')
    items = data.readlines()
    print (len(items))
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
class ModelTwo():
    def __init__(self, data, wordDict):
        self.data = data
        self.dict = wordDict
    def abstractRepresentation(self):
        self.abstract = []
        label = int(self.data[0])
        review_text = self.data[1]
        self.abstract.append([0,label, review_text])
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
                    else:
                        features[form[word]] += 1
            for key in features:
                if features[key] >= 4:
                    del features[key]
            item.append(features)

        self.abstract = np.asarray(self.abstract)
        self.abstract = np.delete(self.abstract, 2, 1)
        return self.abstract
inputFilename = 'smalldata/test_data.tsv'
dict_input = 'dict.txt'
data = loadData(inputFilename)
""" wordDict = representDict(dict_input)
model = ModelTwo(data, wordDict)
abstract = model.denseFeatureRepresentation()
 """