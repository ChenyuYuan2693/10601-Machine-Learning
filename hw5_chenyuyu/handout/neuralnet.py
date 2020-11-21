import numpy as np
import math
import sys

def openFile(inputFile):
    with open(inputFile, 'rt') as raw_data:
        read_csv = np.loadtxt(raw_data, delimiter = ',')
        label = read_csv[:,0]
        data = read_csv[:,1:]
    return label, data
def getDim(matrix):
    nrow = np.shape(matrix)[0]
    ncol = np.shape(matrix)[1]
    return nrow, ncol

def linearForward(target, weight):
    bias_term = weight[:, 0]
    a = bias_term + np.dot(weight[:,1:], target)
    return a

def sigmoidForward(a):
    z = 1/(1+np.exp(-a))
    return z

def linearForward_B(target, weight):
    b = np.dot(weight, target)
    return b

def softmaxForward(b):
    denominator = np.sum(np.exp(b))
    numerator = np.exp(b)
    y_hat = numerator / denominator
    return y_hat

def crossEntropyForward(y, y_hat):
    right_side = np.array([np.log(y_hat)])
    left_side = y
    l = -np.sum(left_side.T*right_side)
    return l


def crossAndSoftmaxBackward(y, b, y_hat):
    return (y_hat - y.T)

def linearBackward_B(z, b, g_b, beta):
    g_beta = np.dot(np.array([z]).T, g_b)
    g_z = np.dot((beta[:,1:]).T, g_b.T)
    return g_beta.T, g_z

def sigmoidBackward(a, z, g_z):
    g_a = g_z.T*(z)*((1-z))
    return g_a

def linearBackward(x, a, g_a):
    g_alpha = np.dot(g_a.T, np.array([x]))
    return g_alpha

def crossEntropy_train(o, outputs):
    return o.J

def crossEntropy_test(alpha, beta, label, data, outputs):
    J = 0
    for d, l in zip(data, label):
        y = np.zeros([outputs, 1])
        y[int(l)] = 1
        a = linearForward(d, alpha)
        z = sigmoidForward(a)
        z = np.insert(z, 0, values = 1)
        b = linearForward_B(z, beta)
        z = np.delete(z, 0)
        y_hat = softmaxForward(b)
        J += crossEntropyForward(y, y_hat)
    J = J / np.shape(data)[0]
    return J

class NNRecorder:
    def __init__(self, x, a, b, z, y_hat, J, alpha, beta):
        self.x = x
        self.a = a
        self.b = b
        self.z = z
        self.y_hat = y_hat
        self.J = J
        self.x_star = np.insert(x, 0, values = 1)
        self.z_star = np.insert(z, 0, values = 1)
        self.alpha = alpha
        self.beta = beta
    
class SGD_Train:
    def __init__(self, input_label, input_data, input_label_test, input_data_test, hidden_units, init_flag, num_epoch, learning_rate):
        self.input_label = input_label
        self.input_data = input_data
        self.input_label_test = input_label_test
        self.input_data_test = input_data_test
        self.hidden_units = hidden_units
        self.init_flag = init_flag
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.outputs = 10
        self.num_examples, self.features = getDim(self.input_data)
    def initial_matrix(self):
        if self.init_flag == 1:
            alpha, beta = self.uniform_weights(self.features, self.hidden_units, self.outputs)
        elif self.init_flag == 2:
            alpha, beta = self.zero_weights(self.features, self.hidden_units, self.outputs)
        return alpha, beta
    def train(self):
        alpha, beta = self.initial_matrix()
        alpha_addition = np.zeros([self.hidden_units, 1])
        beta_addition = np.zeros([self.outputs, 1])
        self.alpha = np.column_stack((alpha_addition, alpha))
        self.beta = np.column_stack((beta_addition, beta))
        train_crossentropy_list = []
        test_crossentropy_list = []
        for i in range(0, self.num_epoch):
            train_mean_crossentropy = 0
            for l, d in zip(self.input_label, self.input_data):
                y = np.zeros([self.outputs, 1])
                y[int(l)] = 1
                self.o = self.NNForward(d, y, self.alpha, self.beta)
                g_alpha, g_beta = self.NNBackward(d, y, self.alpha, self.beta, self.o)
                self.alpha = self.alpha - self.learning_rate*g_alpha
                self.beta = self.beta - self.learning_rate*g_beta
        train_crossentropy = crossEntropy_test(self.alpha, self.beta, self.input_label, self.input_data, self.outputs)
        test_crossentropy = crossEntropy_test(self.alpha, self.beta, self.input_label_test, self.input_data_test, self.outputs)
        return self.alpha, self.beta, train_crossentropy, test_crossentropy
    def NNForward(self, x, y, alpha, beta):
        a = linearForward(x, alpha)
        z = sigmoidForward(a)
        z = np.insert(z, 0, values = 1)
        b = linearForward_B(z, beta)
        z = np.delete(z, 0)
        y_hat = softmaxForward(b)
        J = crossEntropyForward(y, y_hat)
        o = NNRecorder(x, a, b, z, y_hat, J, alpha, beta)
        return o
    
    def NNBackward(self, x, y, alpha, beta, o):
        g_J = 1
        g_b = crossAndSoftmaxBackward(y, o.b, o.y_hat)
        g_beta, g_z = linearBackward_B(o.z_star, o.b, g_b, beta)
        g_a = sigmoidBackward(o.a, o.z, g_z)
        g_alpha = linearBackward(np.insert(x, 0, values = 1), o.a, g_a)
        return g_alpha, g_beta
    
    def uniform_weights(self, features, hidden_units, outputs):
        self.alpha = np.random.uniform(low = -0.1, high = 0.1, size = (hidden_units,features))
        self.beta = np.random.uniform(low = -0.1, high = 0.1, size = (outputs,hidden_units))
        return self.alpha, self.beta
    def zero_weights(self, features, hidden_units, outputs):
        alpha = np.zeros((hidden_units, features), dtype = float)
        beta = np.zeros((outputs, hidden_units), dtype = float)
        return alpha, beta

class SGD_Test:
    def __init__(self, alpha, beta, label, data):
        self.label = label
        self.data = data
        self.alpha = alpha
        self.beta = beta
    def predict(self):
        index_list = []
        for d in self.data:
            a = linearForward(d, self.alpha)
            z = sigmoidForward(a)
            b = linearForward(z, self.beta)
            y_hat = softmaxForward(b)
            argmax = np.max(y_hat)
            index = int(np.argwhere(y_hat == np.max(y_hat)))
            index_list.append(index)
        return index_list
class Outcome:
    def __init__(self, train_list, test_list, train_celist, test_celist, error_train, error_test):
        self.train_list = train_list
        self.test_list = test_list
        self.train_celist = train_celist
        self.test_celist = test_celist
        self.error_train = error_train
        self.error_test = error_test

def getOutcome(alpha, beta, label_train, data_train, label_test, data_test):
    sgd_train = SGD_Test(alpha, beta, label_train, data_train)
    sgd_test = SGD_Test(alpha, beta, label_test, data_test)
    sgd_train_list = sgd_train.predict()
    sgd_test_list = sgd_test.predict()
    return sgd_train_list, sgd_test_list

def calculateError(label_train, label_test, outcome_train, outcome_test):
    temp1 = 0
    temp2 = 0
    for l, o in zip(label_train, outcome_train):
        if l != o:
            temp1 += 1
    error1 = temp1 / len(label_train)
    for l, o in zip(label_test, outcome_test):
        if l != o:
            temp2 += 1
    error2 = temp2 / len(label_test)
    return error1, error2
    
def writeFile(train_output, test_output, metrics_out, train_celist, test_celist):
    with open (metrics_out, 'w') as f:
        for ce_train in train_celist:
            f.write(f'{ce_train}\n')
    f.close()
    with open (train_output, 'w') as f:
        for ce_test in test_celist:
            f.write(f'{ce_test}\n')
    f.close()
        


def main():
    train_input = 'largeTrain.csv'
    test_input = 'largeTest.csv'
    train_output = 'train_out_1.txt'
    test_output = 'test_out.labels'
    metrics_out = 'test_out_1.txt'
    num_epoch = 100
    num_hidden_units = 50
    init_flag = 1
    learning_rate = 0.001
    label_train, data_train = openFile(train_input)
    label_test, data_test = openFile(test_input)
    train_celist = []
    test_celist = []
    for e in range(num_epoch+1):
        sgd_train = SGD_Train(label_train, data_train, label_test, data_test, num_hidden_units, init_flag, e, learning_rate)
        alpha, beta, train_ce, test_ce= sgd_train.train()
        train_celist.append(train_ce)
        test_celist.append(test_ce)
        print (e)
    writeFile(train_output, test_output, metrics_out, train_celist, test_celist)



if __name__ == '__main__':
    main()