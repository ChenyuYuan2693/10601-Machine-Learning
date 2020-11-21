import sys
import csv
import numpy as np
from math import log, exp


class forward_object(object):
	def __init__(self, x = None, a = None, z = None, b = None, y_hat = None, J = None):
		self.x = x
		self.a = a
		self.z = z
		self.b = b
		self.y_hat = y_hat
		self.J = J


def sigmoid(x):
    return (1.0 / (1.0 + exp(-x)))


def init_weights(init_flag, hidden_units, len_of_x, num_of_class):
	if init_flag == 1:
		# init random
		alpha = np.random.uniform(-0.1, 0.1, (hidden_units, len_of_x))
		alpha[:, 0] = 0.0
		beta = np.random.uniform(-0.1, 0.1, (num_of_class, hidden_units + 1)) #plus 1 bias term
		beta[:, 0] = 0.0
		return alpha, beta
	else:
		assert(init_flag == 2)
		# init zeros
		alpha = np.zeros((hidden_units, len_of_x))
		beta = np.zeros((num_of_class, hidden_units + 1)) #plus 1 bias term
		return alpha, beta


def load_data(input_file):
    feature_lst = []
    labels = []
    res_labels = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(int(row[0]))
            tmp_lst = []
            for feature in row[1:]:
            	tmp_lst.append(int(feature))
            features_vec = np.array(tmp_lst)
            features_vec = np.append(1, features_vec) # add x0 = 1
            feature_lst.append(features_vec)
    for i in range(len(labels)):
    	label = labels[i]
    	y = np.zeros(10)
    	y[label] = 1
    	res_labels.append(y)
    return np.array(feature_lst), np.array(res_labels)


def calculate_len_of_vec(data):
	len_of_vec = 0
	for item in data:
		len_of_vec = len(item)
		break
	return len_of_vec


def linear_forward(alpha, training_feature):
	return np.dot(alpha, training_feature)


def sigmoid_forward(a):
	z = np.array([1])  # z0 = 1
	for aj in a.flat:
		zi = sigmoid(aj)
		z = np.append(z,zi)
	return np.array([z]).transpose()


def softmax(bk, b):
	summation = 0.0
	for item in b:
		summation += exp(item)
	res = (exp(bk) / summation)
	return res


def softmax_forward(b):
	y_hat = np.array([])
	for bk in b:
		yi_hat = softmax(bk, b)
		y_hat = np.append(y_hat, yi_hat)
	return np.array([y_hat]).transpose()


def cross_entropy_forward(y, y_hat):
	y_hat_log = np.array([])
	for yi_hat in y_hat:
		y_hat_log = np.append(y_hat_log, log(yi_hat))
	y_hat_log = np.array([y_hat_log]).transpose()
	res = -1 * (np.dot(y.transpose(), y_hat_log))
	return res[0][0]


def softmax_backward(y, y_hat):
	g_b = np.array([])
	for i in range(len(y)):
		g_b = np.append(g_b, float(y_hat[i] - y[i]))
	g_b = np.array([g_b]).transpose()
	return g_b


def linear_backward(g_b, z):
	z_T = z.transpose()
	return np.dot(g_b, z_T)


def linear_backward2(beta_star_T, g_b):
	return np.dot(beta_star_T, g_b)


def sigmoid_backward(g_z, z):
	z = np.delete(z, 0, 0)
	dl_da = z * (1 - z)
	return g_z * dl_da


def linear_backward3(g_a, x_T):
	return np.dot(g_a, x_T)


def NN_forward(training_feature, train_label, alpha, beta, res_of_forward):
	y = train_label
	res_of_forward.a = linear_forward(alpha, training_feature)
	res_of_forward.z = sigmoid_forward(res_of_forward.a)
	res_of_forward.b = linear_forward(beta, res_of_forward.z)
	res_of_forward.y_hat = softmax_forward(res_of_forward.b)
	res_of_forward.J = cross_entropy_forward(y, res_of_forward.y_hat)
	res_of_forward.x = training_feature
	return res_of_forward


def NN_backward(training_feature, train_label, alpha, beta, res_of_forward):
	x = training_feature
	y = train_label
	g_b = softmax_backward(y, res_of_forward.y_hat)
	g_beta = linear_backward(g_b, res_of_forward.z)
	beta_star = np.delete(beta, 0, 1)
	beta_star_T = beta_star.transpose()
	g_z = linear_backward2(beta_star_T, g_b)
	g_a = sigmoid_backward(g_z, res_of_forward.z)
	x_T = res_of_forward.x.transpose()
	g_alpha = linear_backward3(g_a, x_T)
	return g_alpha, g_beta


def mean_cross_entropy(data, labels, alpha, beta):
	entropy_sum = 0.0
	N = len(labels)
	init_forward = forward_object()
	for feature, label in zip(data, labels):
		feature = np.array([feature]).transpose() # init x col vec
		label = np.array([label]).transpose() #init y col vec
		J = NN_forward(feature, label, alpha, beta, init_forward).J
		entropy_sum += J
	return (1 / N) * entropy_sum


def train_NN_SGD(train_data, train_labels, test_data, test_labels, \
	             alpha, beta, num_epoch, learning_rate):
	metrics_str = ""
	init_forward = forward_object()
	i = 0
	j = 0
	for epoch in range(num_epoch):
		for training_feature, train_label in zip(train_data, train_labels):
			training_feature = np.array([training_feature]).transpose() # init x col vec
			train_label = np.array([train_label]).transpose() #init y col vec
			res_of_forward = NN_forward(training_feature, train_label, \
				                        alpha, beta, init_forward)
			gradient_of_alpha, gradient_of_beta = NN_backward(training_feature, train_label, \
			                                                  alpha, beta, res_of_forward)
			alpha = alpha - (learning_rate * gradient_of_alpha)
			beta = beta - (learning_rate * gradient_of_beta)
		train_entropy = mean_cross_entropy(train_data, train_labels, alpha, beta)
		print (i, 'i')
		i += 1
		metrics_str += "epoch=" + str(epoch + 1) + " " + "crossentropy(train):" + " " + str(train_entropy) + "\n"
		test_entropy = mean_cross_entropy(test_data, test_labels, alpha, beta)
		print(j, 'j')
		j+=1
		metrics_str += "epoch=" + str(epoch + 1) + " " + "crossentropy(test):" + " " + str(test_entropy) + "\n"
	return alpha, beta, metrics_str


def write_metrics_out(metrics_out, str_output):
    with open(metrics_out, 'w') as f:
        f.write(str_output)


def write_predict_labels(predictions, output):
    res = ""
    for pred in predictions:
        res += str(pred) + "\n"
    res = res[:-1]
    with open(output, 'w') as f:
        f.write(res)


def NN_prediction(data, labels, alpha, beta):
	prediction_labels = []
	error = 0
	init_forward = forward_object()
	for feature, label in zip(data, labels):
		feature = np.array([feature]).transpose() # init x col vec
		label = np.array([label]).transpose() #init y col vec
		y_hat = NN_forward(feature, label, alpha, beta, init_forward).y_hat
		y_h_class = np.argmax(y_hat)
		prediction_labels.append(y_h_class)
	for i in range(len(prediction_labels)):
		if prediction_labels[i] != np.argmax(labels[i]):
			error += 1
	error_rate = error / len(prediction_labels)
	return prediction_labels, error_rate


def main():
	train_input = 'handout/smallTrain.csv'
	test_input = 'handout/smallTest.csv'
	train_output = 'train_out.labels'
	test_output = 'test_out.labels'
	metrics_out = 'metrics_out.txt'
	num_epoch = 10
	num_hidden_units = 100
	init_flag = 2
	learning_rate = 0.2
	train_data, train_labels = load_data(train_input)
	test_data, test_labels = load_data(test_input)
	len_of_trainx = calculate_len_of_vec(train_data)
	len_of_testx = calculate_len_of_vec(test_data)
	num_of_class = 10
	alpha, beta = init_weights(init_flag, num_hidden_units, len_of_trainx, num_of_class)
	learned_alpha, learned_beta, metrics_str = train_NN_SGD(train_data, train_labels, \
		                                       test_data, test_labels, alpha, \
		                                       beta, num_epoch, learning_rate)
	train_out_labels, error_rate_train = NN_prediction(train_data, train_labels, learned_alpha, learned_beta)
	test_out_labels, error_rate_test = NN_prediction(test_data, test_labels, learned_alpha, learned_beta)
	metrics_str += "error(train): " + str(error_rate_train) + "\n"
	metrics_str += "error(test): " + str(error_rate_test)
	write_metrics_out(metrics_out, metrics_str)
	write_predict_labels(train_out_labels, train_output)
	write_predict_labels(test_out_labels, test_output)



if __name__ == '__main__':
	main()



