import sys
import numpy as np
import math

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
        
def calculatePrior(sentences, tags_list):
    occurence = [0]*len(tags_list)
    occurence = np.asarray(occurence)
    for sentence in sentences:
        word, tag = sentence[0]
        location = tags_list[tag] - 1
        occurence[location] += 1
    return occurence.tolist()

def calculateTransmission(sentences, tags_list):
    occurence = [([0]*len(tags_list))]*len(tags_list)
    occurence = np.asarray(occurence)
    for sentence in sentences:
        for i in range(0, len(sentence)-1):
            previous_tag = sentence[i][1]
            current_tag = sentence[i+1][1]
            previous_location = tags_list[previous_tag] - 1
            current_location = tags_list[current_tag] - 1
            occurence[previous_location][current_location] = occurence[previous_location][current_location]+1
    return occurence.tolist()

def calculateEmission(sentences, tags_list, words_list):
    occurence = [([0]*len(words_list))]*len(tags_list)
    occurence = np.asarray(occurence)
    for sentence in sentences:
        for word in sentence:
            word_location = words_list[word[0]] - 1
            tag_location = tags_list[word[1]] - 1
            occurence[tag_location][word_location] += 1
    return occurence.tolist()

def pseudoCount(matrix):
    matrix = np.asarray(matrix)
    matrix = matrix+1
    return matrix.tolist()  

def normalize_prior(matrix):
    matrix = np.asarray(matrix)
    total = np.sum(matrix)
    return (matrix / total).tolist()

def normalize(matrix):
    matrix = np.asarray(matrix)
    total = np.sum(matrix, axis = 1)
    return np.transpose(np.divide(np.transpose(matrix), total)).tolist()

def writefile(matrix, outputfile):
    if type(matrix[0]) != list:
        write = ''
        for num in matrix:
            write = write + '%.18e'%num + '\n'
    else:
        write = ''
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                write = write + '%.18e'%matrix[i][j]+' '
            write += '\n'
    f = open(outputfile, 'w')
    f.write(write)

def main():
    train_input = 'trainwords.txt'  #sys.argv[1]
    index_to_word = 'index_to_word.txt' #'toy_data/toy_index_to_word.txt' #sys.argv[2]
    index_to_tag = 'index_to_tag.txt' #'toy_data/toy_index_to_tag.txt' #sys.argv[3]
    hmmprior = 'hmmprior_1.txt' #sys.argv[4]
    hmmemit = 'hmmemit_1.txt' #sys.argv[5]
    hmmtrans = 'hmmtrans_1.txt' #sys.argv[6]
    split = load_data_words(train_input)[0:10000]
    print (len(split))
    words_list = load_data_index_to_words(index_to_word)
    tags_list = load_data_index_to_words(index_to_tag)
    prior_matrix = calculatePrior(split, tags_list)
    transmission_matrix = calculateTransmission(split, tags_list)
    emission_matrix = calculateEmission(split, tags_list, words_list)
    prior_matrix = pseudoCount(prior_matrix)
    transmission_matrix = pseudoCount(transmission_matrix)
    emission_matrix = pseudoCount(emission_matrix)
    prior_matrix = normalize_prior(prior_matrix)
    transmission_matrix = normalize(transmission_matrix)
    emission_matrix = normalize(emission_matrix)
    writefile(prior_matrix, hmmprior)
    writefile(transmission_matrix, hmmtrans)
    writefile(emission_matrix, hmmemit)

if __name__ == '__main__':
	main()