from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import time, shutil, os

#parameter
learning_rate = 0.01
training_epochs = 60
batch_size = 100
display_step = 1


def layer(input, w_shape, b_shape):
     #random_normal_initializer:generate tensor with normal distribution
     weight_stddev = (2.0/w_shape[0]) **0.5
     w_init = tf.random_normal_initializer(stddev=weight_stddev)
     bias_init = tf.constant_initializer(value=0)

     w = tf.get_variable("w", w_shape, initializer = w_init)
     b = tf.get_variable("b", b_shape, initializer = bias_init)
     
     return tf.nn.relu(tf.matmul(input, w) + b)
     
def inference(x):

     #2 hidden layer 
     with tf.variable_scope("hidden_1"):
          hidden_1 = layer(x, [784, 256], [256])

     with tf.variable_scope("hidden_2"):
          hidden_2 = layer(hidden_1, [256, 256], [256])

     with tf.variable_scope("output"):
          output = layer(hidden_2, [256, 10], [10])
     return output



def loss(output, y):
     xentropy = xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)  

     loss = tf.reduce_mean(xentropy)
     print(loss)
     return loss

def train(cost, global_step):
     #summary record
     tf.summary.scalar("cost", cost)

     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
     train_op = optimizer.minimize(cost, global_step = global_step)

     return train_op


def evaluate(output, y):
     #計算accuracy
     correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
     ac = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     
     #record
     tf.summary.scalar("error", (1.0 - ac))
     return ac
     
if __name__ =='__main__':
     with tf.Graph().as_default():
          # x y 數據拿到的資料
          x = tf.placeholder("float", [None, 784])
          y = tf.placeholder("float", [None, 10])

          output = inference(x)
          
          cost = loss(output, y)

          global_step = tf.Variable(0, name='global_step', trainable=False)

          train_op = train(cost, global_step)

          eval_op = evaluate(output, y)

          summary_op = tf.summary.merge_all()

          #save
          saver = tf.train.Saver()

          init_op = tf.global_variables_initializer()

          
          with tf.Session() as sess:
               sess.run(init_op)
               summary_writer = tf.summary.FileWriter("logistic_logs/",graph_def=sess.graph_def)
               for epoch in range(training_epochs):
                    avg_cost = 0
                    total_batch = int(mnist.train.num_examples/batch_size)

                    for i in range(total_batch):
                         batch_x, batch_y = mnist.train.next_batch(batch_size)

                         sess.run(train_op, feed_dict = {x:batch_x, y:batch_y})
                         avg_cost = (avg_cost + sess.run(cost, feed_dict = {x:batch_x, y:batch_y})) / total_batch

                    # display

                    if epoch % display_step == 0:
                         print("epoch:", '%04d' % (epoch+1), "cost = ", "{:.9f}".format(avg_cost))

                         ac = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
                         print("ac",ac)

                         summary_str = sess.run(summary_op, feed_dict={x: batch_x, y: batch_y})
                         summary_writer.add_summary(summary_str, sess.run(global_step))
                         saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)


               print("finish")
               accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})
               print("Test Accuracy:", accuracy)  


