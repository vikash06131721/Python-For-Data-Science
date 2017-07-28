try:
    import tensorflow as tf 
    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print ("One or Many packages not installed")

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

#As we all know the features about the   MNIST data
print "Training Images--",len(mnist.train.images)
print "Training Labels--", len(mnist.train.labels)
print "Test Images--", len(mnist.test.images)
print "Test Labels--", len(mnist.test.labels)
print "Validation Images--", len(mnist.validation.images)
print  "Validation Labels--", len(mnist.validation.labels)

#Training
X_train= mnist.train.images
y_train= mnist.train.labels.astype('int')
#Testing
X_test= mnist.test.images
y_test= mnist.test.labels.astype('int')
#Validation
X_validation= mnist.validation.images
y_validation= mnist.validation.labels.astype('int')
#Number of features
print X_train[0].shape

#Lets have a look at the images of mnist
fig, ax = plt.subplots(nrows=4,ncols=4)#4 rows and 4 columns

i=1
while i<17:
    plt.subplot(4,4,i)
    plt.imshow(X_train[i].reshape(28,28))
    i=i+1
plt.show()

#Construction Phase:
n_inputs= 28*28
n_hidden1= 300
n_hidden2= 100
n_outputs= 10

#Defining placeholders
x= tf.placeholder(dtype=tf.float32, shape=[None,n_inputs],name="X") #the variable used for input, it means we may input any number of images with 784featuures
y= tf.placeholder(dtype=tf.int64,shape=[None],name="Y")

#Have a look at the shapes of x,y
print x.get_shape().as_list()
print y.get_shape().as_list()

#Lets define the Neural Network
def neuron_layer(X,n_neurons,name,activation_fn=None):
    with tf.name_scope(name):
        n_inputs= int(X.get_shape()[1])
        stddev= 2/np.sqrt(n_inputs)
        init= tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        W= tf.Variable(init,name="Weights")
        B= tf.Variable(tf.zeros([n_neurons]),name="Bias")
        Z= tf.matmul(X,W) + B
        if activation_fn is not None:
            return activation_fn(Z)
        else:
            return Z

#SO we have a nice fn to create a neuron layer, lets try and create a deep layer:
with tf.name_scope("dnn"):
    hidden1= neuron_layer(X=x,n_neurons=n_hidden1,name="hidden1",activation_fn=tf.nn.relu) #outputs a matrix of form [None,300]
    hidden2= neuron_layer(X=hidden1,n_neurons=n_hidden2, name="hidden2",activation_fn=tf.nn.relu) #outputs a matrix of form [None,100]
    logits= neuron_layer(X=hidden2, n_neurons=n_outputs,name="output") #outputs a matrix of form [None, 10]

#Lets define a loss
with tf.name_scope("loss"):
    xentropy= tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss= tf.reduce_mean(xentropy,name="loss")

#Training
learning_rate=0.01
with tf.name_scope("train"):
    optimizer= tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope("eval"):
    correct= tf.nn.in_top_k(logits,y,1)
    accuracy= tf.reduce_mean(tf.cast(correct,tf.float32))


init= tf.initialize_all_variables()
saver= tf.train.Saver()

n_epochs=40
batch_size=50
num_batches= mnist.train.num_examples//batch_size
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(num_batches):
            X_batch,y_batch=mnist.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={x:X_batch,y:y_batch})
        acc_train= accuracy.eval(feed_dict={x:X_batch,y:y_batch})
        acc_test= accuracy.eval(feed_dict={x: X_test,y:y_test})

        print (epoch, "Train accuracy:",acc_train, "Test accuracy",acc_test)
    save_path= saver.save(sess, "./model.ckpt")
sess.close()
#Testing the model

with tf.Session() as sess:
    saver.restore(sess,'./model.ckpt')
    X_new_scaled = mnist.validation.images[:20]
    Z = logits.eval(feed_dict={x: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

print("Predicted classes:", y_pred)
print("Actual classes:   ", mnist.validation.labels[:20])
