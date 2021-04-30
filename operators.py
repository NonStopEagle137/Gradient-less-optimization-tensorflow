import tensorflow as tf

class operators__:
   def __init__(BASE_MODEL_LAYERS):
     pass
   def _make_non_zero(self, A):
       for i, layer in enumerate(A):
           A[i].assign(tf.where( tf.equal(tf.zeros(layer.shape), tf.Variable(layer)),
                         0.01 * tf.ones_like(tf.Variable(layer)), tf.Variable(layer)))
       return A
   
   def _assign(self,A, B): #B to A
       
       for i,layer in enumerate(A):
           
           A[i].assign(tf.Variable(B[i]).read_value())
           
       return A
   
   def _add(self, A, B):
       
       for i,layer in enumerate(A):
           A[i].assign(tf.cast(tf.add(layer,B[i]), tf.float32))
           
       return A
   def _scale_conserving_add(self, A, B): # very important that B be normalised.
       
       for i,layer in enumerate(A):   
           if tf.reduce_sum(layer) != tf.constant(0.0):
             A[i].assign(tf.add(layer,tf.multiply(B[i], layer)))
           else:
             A[i].assign(tf.add(layer, tf.multiply(B[i], 1e3))) # Initial change in bias is big.
           
       return A
   def _subtract(self, A, B):
       
       for i,layer in enumerate(A):
           A[i].assign(tf.cast(tf.subtract(layer,B[i]), tf.float32))
           
       return A
   def _divide(self, A, B):
       
       for i,layer in enumerate(A):
           A[i].assign(tf.cast(tf.math.divide(layer,B[i]), tf.float32))
           
       return A
   def _multiply(self, A, B):
       for i,layer in enumerate(A):
           A[i].assign(tf.cast(tf.divide(layer,B[i]), tf.float32))
       return A

   def _abs_max(self, A):
       
       for i,layer in enumerate(A):
           self.max_ = (tf.maximum(self.max_,(tf.cast(tf.reduce_max(tf.abs(layer)), tf.float32))))
       return self.max_

   def _abs_(self, A):
       
       for i,layer in enumerate(A):
           A[i].assign(tf.cast(tf.abs(layer), tf.float32))    
       return A
   def _add_scalar(self, A, B):
       
       if B.shape != (1,):
          for i,layer in enumerate(A):
              A[i].assign(tf.cast(tf.add(layer,tf.reshape(B, (1,))), tf.float32))
              
       else:
           for i,layer in enumerate(A):
              A[i].assign(tf.Variable(A[i]).assign(tf.cast(tf.add(layer,B), tf.float32)))
              

       return A
   def _divide_scalar(self, A, B, swap = False):
       
       if swap == False:
          for i,layer in enumerate(A):
              A[i].assign(tf.cast(tf.math.divide_no_nan(tf.reshape(B,(1,)),layer), tf.float32))
              
       else:
         for i,layer in enumerate(A):
              A[i].assign(tf.cast(tf.math.divide_no_nan(layer,tf.reshape(B, (1,))), tf.float32))
              
       return A
   def _norm(self, A):
       for i, layer in enumerate(A):
          if tf.reduce_sum(tf.norm(A[i])) != 0:
             A[i].assign(A[i]/ tf.norm(A[i]))
          else:
             A[i].assign(A[i]/ tf.reduce_logsumexp(tf.norm(A[i])))
       return A

   def _multiply_scalar(self, A, B):
       
       if B.shape != (1,):
          for i,layer in enumerate(A):
           
              A[i] = A[i].assign((tf.cast(tf.multiply(layer,tf.reshape(B, (1,))), tf.float32)))
              
       else:
          for i,layer in enumerate(A):
           
              A[i] = A[i].assign(tf.cast(tf.multiply(layer,B), tf.float32))
              
           
       return A
   def _exp(self, A):
       
       for i,layer in enumerate(A):
           A[i] = A[i].assign(tf.cast(tf.exp(layer), tf.float32))
           
       return A
   def _sign(self, A, B):
       for i,layer in enumerate(A):
           B[i].assign(tf.multiply(tf.math.sign(layer.read_value()), -1e-5))
       return B
   def generate_parameters_variable_like(self, A):
       
       for layer in A:
           A[i] = A[i].assign(tf.ones(shape = layer.shape, dtype = layer.dtype))
           
       return A
