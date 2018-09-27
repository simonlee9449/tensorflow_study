import tensorflow as tf
tf.set_random_seed(777) #for reproducibility

x_train = [1, 2, 3]
y_train = [1, 2, 3]

# y = W * x + b

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
#Intializes global variables in the graph
sess.run(tf.global_variables_initializer())

#Fit the line
for step in range(10001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
