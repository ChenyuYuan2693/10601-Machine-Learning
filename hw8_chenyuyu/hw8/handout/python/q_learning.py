from environment import MountainCar
import sys
import numpy as np
import random

def initial(statespace, actionspace):
    weight = np.zeros([statespace, actionspace], dtype = float)
    bias = 0
    return weight, bias

def linear_approx(weight, bias, state):
    product = np.dot(state, weight)
    qvalue = product + bias
    return qvalue

def epsilon_greedy(epsilon, qvalue, actions):
    prob = np.random.uniform(0,1)
    if prob > epsilon:
        qvalue_list = qvalue.tolist()[0]
        optimal_q = max(qvalue_list)
        action = qvalue_list.index(optimal_q)    
    else:
        action = random.choice(actions)
    return action

def gradient_calculation(state, action, weight):
    row, col = weight.shape
    gradient = np.zeros([row, col])
    for i in range(0, row):
        gradient[i, action] = state[0,i]
    return gradient

def statetovalue(car, state):
    result = np.zeros([1, car.state_space])
    for s in state:
        result[0, s] = state[s]
    return result

def train(car, actions, weight, bias, episodes, max_iterations, epsilon, gamma, learning_rate):
    reward_list = []
    for i in range(0, episodes):
        state = car.reset()
        state_value = statetovalue(car, state)
        reward_total = 0
        for i in range(0, max_iterations):
            qvalue = linear_approx(weight, bias, state_value)
            next_action = epsilon_greedy(epsilon, qvalue, actions)
            next_state, reward, done = car.step(next_action)
            reward_total += reward
            next_state_value = statetovalue(car, next_state)
            next_qvalue = linear_approx(weight, bias, next_state_value)
            gradient = gradient_calculation(state_value, next_action, weight)
            TD_error = qvalue[0, next_action] - (reward+gamma*np.max(next_qvalue))
            weight = weight - learning_rate*TD_error*gradient
            bias = bias - learning_rate*TD_error
            state = next_state
            state_value = next_state_value
            if (done == True):
                break
        reward_list.append(reward_total)
    return (weight, bias, reward_list)

def write_weight_out(weight_out, weight, bias):
    content = str(bias)+'\n'
    row, col = weight.shape
    for i in range(row):
        for j in range(col):
            content += str(weight[i][j])+'\n'
    f = open(weight_out, 'w')
    f.write(content)

def write_returns_out(reward, returns_out):
    content = ''
    for i in reward:
        content += str(i)+'\n'
    f = open(returns_out, 'w')
    f.write(content)



def main():
    """ car = MountainCar(sys.argv[1])
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])  """ 
    car = MountainCar('tile')#sys.argv[1])
    weight_out = 'weight_tile.txt' #sys.argv[2]
    returns_out = 'returns_tile.txt' #sys.argv[3]
    episodes = 400 #sys.argv[4]
    max_iterations = 200 #sys.argv[5]
    epsilon = 0.05 #sys.argv[6]
    gamma = 0.999 #sys.argv[7]
    learning_rate = 0.00005 #sys.argv[8]  

    actions = [0, 1, 2]
    weight, bias = initial(car.state_space, len(actions))
    weight, bias, reward_record = train(car, actions, weight, bias, episodes, max_iterations, epsilon, gamma, learning_rate)

    write_weight_out(weight_out, weight, bias)
    write_returns_out(reward_record, returns_out)

if __name__ == "__main__":
    main()