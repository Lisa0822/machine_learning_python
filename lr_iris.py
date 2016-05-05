import pandas as pd
import numpy as np
import random
import theano

print '..........................load data ...........................................'
# load data and split
def load_data():
   iris = pd.read_csv("data/iris.txt",index_col=0)
   label = np.where(iris.Species=='setosa',0,
         np.where(iris.Species=='versicolor',1,
            np.where(iris.Species=='virginica',2,iris.Species)
         )
      )
   #label = map(lambda i: int(i), [i for i in label])
   rate = 0.8
   n = iris.shape[0]
   train_index = random.sample(range(n), int(n*rate)) 
   train_set_x = iris.iloc[train_index,0:-1]
   train_set_y = label[train_index]
   train_set_y = map(lambda i: int(i), [i for i in train_set_y])   

   test_index = list(set(range(n))-set(train_index))
   test_set_x = iris.iloc[test_index,0:-1]
   test_set_y = label[test_index]
   test_set_y = map(lambda i: int(i), [i for i in test_set_y])
   return train_set_x,train_set_y,test_set_x,test_set_y

train_set_x,train_set_y,test_set_x,test_set_y = load_data()

# n_in : number of variables; n_out: number of classes
n_in = train_set_x.shape[1]
n_out = len(set(train_set_y))

# define input and output
x = theano.tensor.matrix('x')
y = theano.tensor.ivector('y')

# initial weights and bias
w = theano.shared(
   value = np.zeros((n_in,n_out),dtype=theano.config.floatX),
   name = 'w',
   borrow = True
)

b = theano.shared(
   value = np.zeros((n_out,),dtype=theano.config.floatX),
   name = 'b',
   borrow = True
)

# calculate p_y and y_pred
p_y = theano.tensor.nnet.softmax(theano.tensor.dot(x,w) + b)
y_pred = theano.tensor.argmax(p_y, axis = 1)

#test
f_y = theano.function(
   inputs = [x],
   outputs = [p_y, y_pred]
   )

#f_y(train_set_x)


# cost function: mean of p of true class 
cost = -theano.tensor.mean(theano.tensor.log(p_y)[theano.tensor.arange(x.shape[0]),y])

f_cost = theano.function(
   inputs = [x,y],
   outputs = cost
   )

g_w = theano.tensor.grad(cost, w)
g_b = theano.tensor.grad(cost, b)

# define learning_rate
learning_rate = 0.05
updates = [(w,w-learning_rate*g_w), (b, b-learning_rate*g_b)]

train_model = theano.function(
   inputs = [x,y],
   outputs = [y_pred, cost],
   updates = updates
   ) 

test_model = theano.function(
   inputs = [x,y],
   outputs = [y_pred, cost] 
   )


print '.......................training model ...................................'
iters = 1000
loop = True

for j in range(iters):
   y_pred, cost = train_model(train_set_x, train_set_y)
   z = zip(train_set_y, y_pred) 
   right_rate = 100.0 * sum(map(lambda i: int(np.where(z[i][0]==z[i][1],1,0)), [i for i in range(len(z))]))  / len(z) 
   print 'iters = %d, cost = %.2f, right_rate = %.2f%%' %(j, cost, right_rate)
#print w.get_value(), b.get_value()
#print y_pred

print '.......................test model.......................................'
y_pred, cost = test_model(test_set_x, test_set_y)
z = zip(test_set_y, y_pred)
right_rate = 100.0 * sum(map(lambda i: int(np.where(z[i][0]==z[i][1],1,0)), [i for i in range(len(z))]))  / len(z)
print 'The final result is: cost = %.2f, right_rate = %.2f%%' %(cost, right_rate)
