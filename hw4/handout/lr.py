import numpy as np
import sys
import math
def loadData(input):
    data = open(input, 'r')
    items = data.readlines()
    data.close()
    label = []
    features = []
    for item in items:
        word = item.split('\t')
        label.append(word[0])
        row = []
        for indexword in word[1:]:
            index = int(indexword.split(':')[0])
            row.append(index)
        features.append(row)
    return label, features
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
def sparse_dot(X, W):
    product = 0.0
    for x in X:
        product += W[x]
    return product+W[-1]
def SGD_update(features, label, wordDict, W):
    learning_rate = 0.1
    for f, l in zip(features, label):
        l = float(l)
        product_dot = sparse_dot(f, W)
        f_vec = np.zeros(len(wordDict)+1)
        for index in f:
            f_vec[index] = 1.0
        f_vec[-1] = 1.0
        neg = math.exp(product_dot)/(1+math.exp(product_dot))
        W = W - learning_rate*((neg-l)*f_vec)    
    return W
def train(features, label, wordDict, epochs):
    dim = len(wordDict)+1
    W = np.zeros(dim)
    for i in range(epochs):
        W = SGD_update(features, label, wordDict, W)
    return W
def test(features, labels, W):
    predList = []
    for f in features:
        product_dot = sparse_dot(f, W)
        prob = np.exp(product_dot)/(1+np.exp(product_dot))
        if prob >= 0.5:
            pred = 1
        else:
            pred = 0
        predList.append(pred)
    return predList
def compare(pred, label):
    false = 0
    for i in range(len(pred)):
        if int(pred[i]) != int(label[i]):
            false += 1
    error = false/len(pred)
    return error
        
def run(inputfile_train, inputfile_test, outputfile_train, outputfile_test, metrics_out, dict_input, epoch):
    train_label, train_features = loadData(inputfile_train)
    test_label, test_features = loadData(inputfile_test)
    wordDict = representDict(dict_input)
    W = train(train_features, train_label, wordDict, epoch)
    predList_train = test(train_features, train_label, W)
    predList_test = test(test_features, test_label, W)
    with open(outputfile_train, 'w') as f:
        for p in predList_train:
            f.write(str(p)+'\n')
    f.close()
    with open(outputfile_test, 'w') as f:
        for p in predList_test:
            f.write(str(p)+'\n')
    f.close() 
    error_train = compare(predList_train, train_label)
    error_test = compare(predList_test, test_label)
    with open(metrics_out, 'w') as f:
        f.write(f'error(train): {error_train}\n')
        f.write(f'error(test): {error_test}\n')
    f.close() 
    print (error_test)
def main():
    formatted_train_input = 'formatted_train.tsv'#sys.argv[1]
    formatted_valid_input = 'formatted_valid.tsv'#sys.argv[2]
    formatted_test_input = 'formatted_test.tsv'#sys.argv[3]
    dict_input = 'dict.txt'#sys.argv[4]
    train_out = 'neg_train.txt'#sys.argv[5]
    test_out = 'neg_valid.txt'#sys.argv[6]
    metrics_out = '2.txt'#sys.argv[7]
    num_epoch = 30 #int(sys.argv[8])
    run(formatted_train_input, formatted_test_input, train_out, test_out, metrics_out, dict_input, num_epoch)

if __name__ == '__main__':
    main()