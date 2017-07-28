import tensorflow as tf
import numpy as np
import pandas as pd

#Now linear regression is the simplest of the models:
#Y= X*W +B, i.e you put the value of X, W and B and you get Y.

#Lets consider a simple linear regression data
data_file= './tensorflow_data/fire_theft.xls'

data= pd.read_excel(data_file)
train= data[:30]
#The general flow for building a tensorflow model:
#Step 1: Set up your placeholders
X_plceholder= tf.placeholder(dtype=tf.float32,name="X")
Y_placeholder= tf.placeholder(dtype=tf.float32,name="Y")

#Step2: Define Weights and biases
w= tf.Variable(0.0,name="weights")
b= tf.Variable(0.0,name="biases")


#Step3: Define the model
#y = x*w + b
y_pred1= tf.multiply(X_plceholder,w)
y_pred=tf.add(y_pred1,b,name="prediction")
#oUr objective is to calculate the appropriate weights and biases

#step 4: For saving the graph aka session
saver= tf.train.Saver()

#Step 5: computing the loss
loss= tf.square(y_pred-Y_placeholder,name="loss")

#Step 6: minimize the loss bygradient descent
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

#Step7: Run the whole damn thing a thousand times !!

with tf.Session() as sess:
    #Initialize all the variables we are goin to use
    sess.run(tf.initialize_all_variables())

    for i in xrange(1000):
        loss_list=[]
        total_loss=0
        for x,y in train.values:
            _,l= sess.run([optimizer,loss],feed_dict={X_plceholder:x,Y_placeholder:y})
            total_loss= total_loss+l
            loss_list.append(total_loss)
        print "mean loss:",np.mean(loss_list)
        saver.save(sess,"linear_reg",global_step=1000)

    print sess.run([w,b])

sess.close()
