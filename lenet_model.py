import tensorflow as tf
from utils import  *
from sklearn.utils import shuffle
import os


class LeNetModel():
    def __init__(self, logits, x_train, y_train, x_valid, y_valid,  dir, learning_rate, x, y):
        self.logits = logits
        self.x_train, self.x_valid = x_train, x_valid
        self.y_train, self.y_valid = y_train, y_valid
        self.x = x
        self.y = y
        self.one_hot_y = tf.one_hot(self.y, 43)
        self.learning_rate = learning_rate
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir = dir
        self.save_path = os.path.join(dir, 'best_validation')
        
    # def evaluate(self, x_data, y_data):
    #     num_examples = len(x_data)
    #     total_accuracy = 0
    #     sess = tf.get_default_session()
    #

    def evaluation_operation(self):
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
            eval_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", eval_operation)
        return eval_operation
        
    def evaluate(self, X_data, y_data, eval_op, summ, writer, i):
        
        BATCH_SIZE = 3
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
            acc, summary = sess.run([eval_op, summ], feed_dict={ self.x: batch_x, self.y: batch_y})
            total_accuracy += (acc * len(batch_x))
        writer.add_summary(summary, i)
        return total_accuracy / num_examples

    def training_operation(self):
        
        with tf.name_scope("cross_ent"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y,
                                                                logits=self.logits)
            loss_operation = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("cross_ent", loss_operation)
        
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            training_operation = optimizer.minimize(loss_operation)
        
        return training_operation

    def train(self, epochs, batch_size):
        training_operation = self.training_operation()
        eval_op = self.evaluation_operation()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        best_validation_accuracy = 0
        # define the training operation
        summ = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(init)
            num_examples = len(self.x_train)
            
            writer = tf.summary.FileWriter('./output', sess.graph)
            writer.add_graph(sess.graph)
            for i in range(epochs):
                self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = self.x_train[offset:end], self.y_train[offset:end]
                    _, summary = sess.run([training_operation, summ], feed_dict={self.x: batch_x, self.y: batch_y})
                    #writer.add_summary(summary, offset+1)
                
                validation_accuracy = self.evaluate(self.x_valid, self.y_valid, eval_op, summ, writer, i)
                if (validation_accuracy > best_validation_accuracy):
                    improvment_msg = 'Improved from {} to {}'.format(best_validation_accuracy, validation_accuracy)
                    # Update the best-known validation accuracy.
                    best_validation_accuracy = validation_accuracy
    
                    # Set the iteration for the last improvement to current.
                    last_improvement = i
                    print(best_validation_accuracy)
    
                    # Save all variables of the TensorFlow graph to file.
                    saver.save(sess=sess, save_path=self.save_path)
    
                    print('EPOCH {} ...'.format(i+1))
                    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                    print(improvment_msg)
                    print()
                else:
                    print('EPOCH {} ...'.format(i + 1))
                    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                    print('did not improve')
                    print()
            writer.close()
            
            #saver.save(sess, self.dir)
            print('Model saved')
            
            
            
    def predict(self, y):
        return tf.equal(tf.argmax(self.logits, 1), tf.argmax(y, 1))
    
    
    

    
    