import random
import sys

import numpy as np

 
a = 0.5*0.9*0.1*0.9
print (a)

b = 0.9*0.45*0.1
c = 0.9*0.25*0.9
d = 0.5*0.45*0.8
e = 0.5*0.25*0.2
print (b, c, d, e)

w = np.asarray([[7,2], [4,1], [1,4]])
print (w)
h = np.asarray([[1,5,8], [3,4,1]])

r = np.dot(w, h)
print (r)

print (25/4)