import tensorflow as tf
from utils import  *
from sklearn.utils import shuffle
import os
 
MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR):
    os.makedirs(dir)

class LeNetModel():
    
    def __init__(self, logits, x_train, y_train, x_valid, y_valid, learning_rate, x, y, holdprob, hparam):
        self.logits = logits
        self.x_train, self.x_valid = x_train, x_valid
        self.y_train, self.y_valid = y_train, y_valid
        self.x = x
        self.y = y
        self.one_hot_y = tf.one_hot(self.y, 43)
        self.learning_rate = learning_rate
        self.save_path = os.path.join(MODEL_DIR, hparam)
        self.hold_prob = holdprob
        self.hparam = hparam

    def evaluation_operation(self):
        '''
        Evaludation metric
        :return:
        '''
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
            eval_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='evaluation_operation')
            tf.summary.scalar("accuracy", eval_operation)
        return eval_operation
    
    def loss_operation(self):
        with tf.name_scope("xent_loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y,
                                                                    logits=self.logits)
            loss_operation = tf.reduce_mean(cross_entropy)
        return loss_operation
    

    def training_operation(self):
        loss_operation = self.loss_operation()
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            training_operation = optimizer.minimize(loss_operation)
        
        return training_operation

    def train(self, epochs, batch_size):
        
        training_operation = self.training_operation()
        eval_op = self.evaluation_operation()
        loss_op = self.loss_operation()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        best_validation_accuracy = 0
        
        training_summary = tf.summary.scalar("training_accuracy", eval_op)
        validation_summary = tf.summary.scalar("validation_accuracy", eval_op)
        training_loss_summary = tf.summary.scalar("training_loss", loss_op)
        validation_loss_summary = tf.summary.scalar("validation_loss", loss_op)

        
        
        #summ = tf.summary.merge_all()
        best_validation_accuracy = 0
        best_training_accuracy = 0
        
        with tf.Session() as sess:
            sess.run(init)
            num_examples = len(self.x_train)
            
            writer_val = tf.summary.FileWriter(MODEL_DIR + self.hparam + 'val', sess.graph)
            writer_train = tf.summary.FileWriter(MODEL_DIR + self.hparam + 'train', sess.graph)
            
            writer_train.add_graph(sess.graph)
            writer_val.add_graph(sess.graph)
            
            for i in range(epochs):
                self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = self.x_train[offset:end], self.y_train[offset:end]
                    
                    sess.run(training_operation, feed_dict={self.x: batch_x, self.y: batch_y, self.hold_prob: 0.5})

                training_accuracy, train_summary = sess.run([eval_op, training_summary], feed_dict={self.x: self.x_train[0:1000], self.y: self.y_train[0:1000], self.hold_prob:1.0})
                
                validation_accuracy, validation_summ = sess.run([eval_op, validation_summary], feed_dict={self.x: self.x_valid, self.y: self.y_valid, self.hold_prob:1.0})
                
                train_loss, train_loss_summ = sess.run([loss_op, training_loss_summary], feed_dict={self.x: self.x_train[0:1000], self.y: self.y_train[0:1000], self.hold_prob:0.5})
                
                validation_loss, validation_loss_summ = sess.run([loss_op, validation_loss_summary], feed_dict={self.x: self.x_valid, self.y: self.y_valid, self.hold_prob:0.5})

                writer_train.add_summary(train_summary, i)
                writer_train.add_summary(train_loss_summ, i)
                writer_train.flush()
                
                writer_val.add_summary(validation_summ, i)
                writer_val.add_summary(validation_loss_summ, i)
                
                writer_val.flush()
                
                
                
                if (validation_accuracy > best_validation_accuracy):
                    
                    improvment_msg = 'Improved from {} to {}'.format(best_validation_accuracy, validation_accuracy)
                    best_validation_accuracy = validation_accuracy
                    print(best_validation_accuracy)
    
                    # Save all variables of the TensorFlow graph to file.
                    saver.save(sess=sess, save_path=self.save_path)
    
                    print('EPOCH {} ...'.format(i+1))
                    print("Training Accuracy = {:.3f}".format(training_accuracy))
                    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                    print(improvment_msg)
                    print()
                else:
                    print('EPOCH {} ...'.format(i + 1))
                    print("Training Accuracy = {:.3f}".format(training_accuracy))
                    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                    print('did not improve')
                    print()
            print(best_validation_accuracy)
            
            
            
    def predict(self, y):
        return tf.equal(tf.argmax(self.logits, 1), tf.argmax(y, 1))
    
    
    

    
    
