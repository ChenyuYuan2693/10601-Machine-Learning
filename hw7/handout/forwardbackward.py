import sys
import numpy as np

def load_data_words(inputfile):
    file_input = open(inputfile, 'r')
    sentences = file_input.readlines()
    new_sentences = []
    for sentence in sentences:
        sentence = sentence.replace('\n', '')
        new_sentences.append(sentence)
    new_split = []
    for sentence in new_sentences:
        words = sentence.split(' ')
        words_list = []
        for word in words:
            tag = word.split('_')
            words_list.append(tag)
        new_split.append(words_list)
    return (new_split)
    
def load_data_index_to_words(inputfile):
    file_input = open(inputfile, 'r')
    words = file_input.readlines()
    new_words = []
    for word in words:
        word = word.replace('\n', '')
        new_words.append(word)
    word_dict = dict()
    i = 1
    for word in new_words:
        word_dict[word] = i
        i += 1
    return word_dict

def logsumexp(x, y):
    m = max(x, y)
    total = np.log(1+np.exp(-abs(x-y)))
    return (m+total)

def calculateAlpha(pi, a, b, words_list, tags_list, split):
    alpha_list = []
    for setence in split:
        alpha = np.zeros((len(setence), len(tags_list)))
        for i in range(0, len(setence)):
            word, tag = setence[i]
            word_location = words_list[word] - 1
            if (i == 0):
                alpha_sub = b[:,word_location]+pi
                alpha[i,:] = alpha_sub
            else:
                alpha_prev = np.transpose(alpha[i-1,:])
                for j in range(0, len(tags_list)):
                    a_transpose_sub = a[:,j]
                    matrix_total = a_transpose_sub + alpha_prev
                    total = float('-inf')
                    for k in matrix_total:
                        total = logsumexp(total, k)
                    alpha[i, j] = total + b[j, word_location]
        alpha_list.append(alpha)
    return (alpha_list)

def calculateBeta(pi, a, b, words_list, tags_list, split, alpha):
    beta_list = []
    for setence in split:
        beta = np.zeros((len(setence), len(tags_list)))
        for i in range(len(setence)-1, 0, -1):
            word, tag = setence[i]
            word_location = words_list[word] - 1
            for j in range(0, len(tags_list)):
                total = float('-inf')
                matrix_total = np.transpose(b[:, word_location])+beta[i,:]+a[j,:]
                for k in matrix_total:
                    total = logsumexp(total, k)
                beta[i-1, j] = total
        beta_list.append(beta)
    return(beta_list)

def get_key(dicts, value):
    for k,v in dicts.items():
        if (v == value):
            return k

def predict(pi, a, b, alpha_list, beta_list, split, words_list, tags_list):
    result = []
    for i in range(0, len(split)):
        sentence = split[i]
        alpha = alpha_list[i]
        beta = beta_list[i]
        prediction = alpha+beta
        tags_pred = np.argmax(prediction, axis = 1)+1
        result_sentence = []
        for j in range(0, len(tags_pred)):
            word = sentence[j][0]
            tag = get_key(tags_list, tags_pred[j])
            result_sub = [word, tag]
            result_sentence.append(result_sub)
        result.append(result_sentence)
    return result

def writefile(filename, prediction):
    write = ''
    for sentence in prediction:
        for word in sentence:
            write += (word[0]+'_'+word[1]+' ')
        write = write[:-1]
        write += '\n'
    f = open(filename, 'w')
    f.write(write)

def writeMetric(filename, alpha_list, prediction, predicted_file):
    total = 0
    for alpha in alpha_list:
        alpha_T = alpha[(len(alpha)-1),:]
        logprob_total = float('-inf')
        for i in alpha_T:
            logprob_total = logsumexp(logprob_total, i)
        total += logprob_total
    ave_loglike = total / len(alpha_list)
    count = 0
    total = 0
    for i in range(0, len(predicted_file)):
        actual = predicted_file[i]
        pred = prediction[i]
        for word_act, word_pred in zip(actual, pred):
            total += 1
            if word_act[1] == word_pred[1]:
                count += 1
    accuracy = count/total
    write = f'Average Log-Likelihood: {ave_loglike}\n'
    write += f'Accuracy: {accuracy}'
    f = open(filename, 'w')
    f.write(write)



def main():
    train_input = 'trainwords.txt'
    test_input = 'testwords.txt' #sys.argv[1]
    index_to_word = 'index_to_word.txt' #sys.argv[2]
    index_to_tag = 'index_to_tag.txt' #sys.argv[3]
    hmmprior = 'hmmprior_1.txt' #sys.argv[4]
    hmmemit = 'hmmemit_1.txt' #sys.argv[5]
    hmmtrans = 'hmmtrans_1.txt' #sys.argv[6]
    metric_file = '3.txt' #sys.argv[8]
    metric_file_1 = '4.txt'
    pi = np.loadtxt(hmmprior)
    a = np.loadtxt(hmmtrans)
    b = np.loadtxt(hmmemit)
    pi = np.log(pi)
    a = np.log(a)
    b = np.log(b)
    split_1 = load_data_words(train_input)
    split = load_data_words(test_input)
    words_list = load_data_index_to_words(index_to_word)
    tags_list = load_data_index_to_words(index_to_tag)
    alpha_list = calculateAlpha(pi, a, b, words_list, tags_list, split)
    beta_list = calculateBeta(pi, a, b, words_list, tags_list, split, alpha_list)
    alpha_list_1 = calculateAlpha(pi, a, b, words_list, tags_list, split_1)
    beta_list_1 = calculateBeta(pi, a, b, words_list, tags_list, split_1, alpha_list_1)
    prediction = predict(pi, a, b, alpha_list, beta_list, split, words_list, tags_list)
    prediction_1 = predict(pi, a, b, alpha_list_1, beta_list_1, split_1, words_list, tags_list)
    #writefile(predicted_file, prediction)
    writeMetric(metric_file, alpha_list, prediction, split)
    writeMetric(metric_file_1, alpha_list_1, prediction_1, split_1)


if __name__ == '__main__':
	main()