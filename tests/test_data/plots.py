import matplotlib.pyplot as plt
import random

def linearish(num):
    x_arr = [i + 0.5*random.random() for i in range(1,num+1)]
    y_arr = [5*(i + 5*random.random()) for i in range(1,num+1)]
    return [x_arr, y_arr]

def quadratish(num):
    x_arr = [i + 0.15*random.random() for i in range(1,num+1)]
    y_arr = [i + 3*random.random() for i in range(1,num+1)]
    y_arr = [i*i for i in y_arr]
    return [x_arr, y_arr]
num = 40
linear = linearish(num)
plot = plt.scatter(linear[0], linear[1])

quadratic = quadratish(num)
x,y = quadratic[0], quadratic[1]
plt.scatter(x, y)

plt.show()
