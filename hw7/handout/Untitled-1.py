import numpy as np
import math
a = np.asarray([[1,2,100,4,5], [6,7,8,9,10]])
print (np.argmax(a, axis = 0), np.argmax(a, axis = 1))


pi = np.array([2/5, 3/5])
a = np.array([[1/4, 3/4], [3/5, 2/5]])
b = np.array([[1/6, 2/3, 1/6], [3/8, 1/8, 1/2]])
alpha_1 = b[:,2]*pi
print (alpha_1)
alpha_2 = b[:,1]*(np.dot(np.transpose(a), alpha_1))
print (alpha_2)
alpha_3 = b[:,0]*(np.dot(np.transpose(a), alpha_2))
print (alpha_3)
total = np.sum(alpha_3)
print(total)
print (np.log(total))