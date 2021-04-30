from tensorflow import keras
import numpy as np
import tensorflow as tf
import os
from operators import operators__
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from tensorflow.keras import activations
from tensorflow.python.client import device_lib
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'



physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


class NumericalOptimizer(operators__):
   def __init__(self): # Lets try to build a static graph here to reduce the complexity.
     self.STEP_SIZE = tf.constant([1e-4], tf.float32);
     self.batch_size = 1000;
     pass
   def model(self, model, loss):
     self.model_ = model
     self.prev_state = model # to store the previous state safely.
     self.loss = loss
     self.x = list()
     self.x_moving = list()
     self.momentum_ = list()
     self.delta = list()
     self.copy_var = list()
     self.disturbance  =list() # we make a small constant disturbance to get the momentum direction
     self.local_fitness= tf.Variable(100.1, tf.float32)
     self.fitness_baseline = tf.Variable(100.1, tf.float32)
     self.df = tf.Variable(1.0, tf.float32)
     self.decay = tf.Variable(1.0, tf.float32) # for decaying delta
     self.var1 = tf.Variable(1.0, tf.float32)

     self.weights_ = list()
     self.x_true = tf.Variable(tf.zeros(shape = (self.batch_size,1,8)));
     self.y_true = tf.Variable(tf.zeros(shape = (self.batch_size,1))); # could be a batch too.
     for layer in self.model_.trainable_variables:
       self.x.append(tf.Variable(tf.Variable(layer).read_value())) # this will be our static variable
       self.weights_.append(tf.Variable(tf.Variable(layer).read_value()))
       self.x_moving.append(tf.Variable(tf.Variable(layer).read_value())) # this will be our moving variable
       self.momentum_.append(tf.Variable(tf.Variable(layer).read_value())) # We'll save the momentum here. This is a Variable.
       self.delta.append(tf.Variable(tf.Variable(tf.random.truncated_normal(layer.shape)).read_value())) # This is the delta we add to x/ x_moving.
       self.disturbance.append(tf.Variable(tf.Variable(tf.multiply(tf.ones(layer.shape), 1e-5)).read_value()))
       self.copy_var.append(tf.Variable(tf.Variable(layer).read_value())) # this is a expendible variable for copying values 
                                                                          # around
   def eval_fitness(self, x):
       self.assign_to_model(x);
       self.var1.assign(tf.Variable(tf.reduce_sum(tf.cast(self.loss(self.y_true,self.model_(self.x_true)), tf.float32))))
       return self.var1
   def initialise_delta(self): # at every step the first delta should be a random disturbance.
      for layer in self.model_.trainable_variables:
          self.delta.append(tf.Variable(tf.Variable(tf.random.truncated_normal(layer.shape)).read_value()))
   def assign_to_model(self, new_weights, update_main = False):
        for i,layer in enumerate(self.model_.trainable_variables):
            layer = layer.assign(tf.cast(new_weights[i], tf.float32))
        return self.model_
   def calculate_momentum_based_on_disturbance(self, MMF):
     self.copy_var = self._assign(self.copy_var, self.x)
     self.local_fitness.assign(self.eval_fitness(self._add_scalar(self.copy_var, self.STEP_SIZE)))
     self.df.assign(tf.subtract(self.local_fitness, self.fitness_baseline));
     self.momentum_ = self._norm(self._divide_scalar(self.disturbance, self.df)) # this is actually the normalised nomentum.
     self.decay.assign(tf.constant(-1.0*MMF))
     self.delta = self._multiply_scalar(self.momentum_, self.decay);

   def _apply_dense(self,weights_model,x_true, y_true):
     def true_():
         self.assign_to_model(self.x_moving)
         return 1
     def false_():
         return 0
     self.x_true.assign(x_true)
     self.y_true.assign(y_true)
     self.x = self._assign(self.x, weights_model) # updating x
     self.fitness_baseline.assign(tf.reshape(self.eval_fitness(self.x), self.fitness_baseline.shape))
     self.calculate_momentum_based_on_disturbance(0.00001)
     self.x_moving = self._add(self.x_moving, self.delta) # this is the delta that we got from the momentum_

     self.local_fitness.assign(self.eval_fitness(self.x_moving)) # resuing this variable

     return tf.cond(self.local_fitness < self.fitness_baseline, true_, false_) 
   def train(self, x_train, x_valid, y_train, y_valid):
    
    epochs = 100;
    for self.iter_ in range(epochs):
        loss_train = 0.0;
        loss_test = 0.0;
        t1 = time.time()
        for i,entry in enumerate(x_train):
            
            if i% self.batch_size != 0:
               continue
            self.weights_ = self._assign(self.weights_, self.model_.trainable_variables);

            try:
                Y = y_train[i:i+self.batch_size].reshape(self.batch_size, 1)

                improvement = self._apply_dense(self.weights_,x_train[i:i+self.batch_size], Y);
                
                if improvement == 1:
                    self.prev_state = self.model_
                else:
                    self.model_ = self.prev_state
            
            except:
                continue

            loss_train += self.loss(y_train[i:i+self.batch_size],self.model_(x_train[i:i+self.batch_size])).numpy().flatten().sum()
            loss_test += self.loss(y_valid[i:i+self.batch_size], self.model_(x_valid[i:i+self.batch_size])).numpy().flatten().sum()
        print("===============================")
        print("Iteration Number : ", self.iter_)
        print("Current Train Loss : ", loss_train)
        print("Current Test Loss : ", loss_test)
        print("===============================")
        print("Time for Epoch : ",time.time() - t1)

def main():
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).reshape(len(X_train), 1, 8)
    X_valid_scaled = scaler.transform(X_valid).reshape(len(X_valid), 1, 8)
    X_test_scaled = scaler.transform(X_test).reshape(len(X_test), 1, 8)
    loss = tf.keras.losses.MSE
    initializer = tf.keras.initializers.VarianceScaling(
                    scale=0.1, mode='fan_in', distribution='uniform')

    M= keras.models.Sequential([keras.layers.Dense(19, input_shape=[1,8], kernel_initializer=initializer),
    keras.layers.Dense(75, kernel_initializer=initializer), keras.layers.Dense(1, kernel_initializer=initializer)])
    M.compile(loss=loss)
    Object_ = NumericalOptimizer()
    Object_.model(M, loss)
    Object_.train(X_train_scaled, X_valid_scaled, y_train.reshape(len(y_train), 1), y_valid.reshape(len(y_valid), 1))

if __name__ == '__main__':
    main()
