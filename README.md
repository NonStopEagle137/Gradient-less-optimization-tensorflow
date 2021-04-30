# Gradient-less-optimization-tensorflow
This is a project that aims to make gradient-less Neural Network optimization possible in tensorflow. 

# The Problem I have tried to solve
Currently, tensorflow offers a number of excellent optimsation algorithms. However all of these require the gradient information to work. That is, they first calculate the gradient of the fitness with respect to the weights, and then update the weights of the Neural Network accordingly. In this project I have tried to create a pipeline/ methodology that allows the user to update the weights of the Neural Network without the need of gradient descent based optimisation algorithms. This means, that This repository allows the user to create their own optimisation algorithm and painlessly (almost) integrate it in any model that they're training.

# How It works

1. In this approach, we first strip the trainable parameters from the neural network and pass them to our optimization algorithm.
2. The Optimization algorithm then proceeds to minimize the loss, post which it returns the parameters, which are then applied back to the neural network.
3. We then can evaluate the loss as we normally do.

# Implementation Complexity

1. Unfortunately, Tensorflow tores the trainable variables as a list of layers, each having a different tensor for the weights and biases.
2. This makes the whole process more difficult to implement. Every operation on the parameters has to loop through all the weights and biases and then make updates.
3. For this I have made some common operators that make this process al little less painful.

# Performance

1. Looping over all the layers is computationally expensive. Obviously this is slower than the built in methods. Having said that its not so slow as to inhibit its use.
2. One thing to note is that the speed is good for larger tensors, as the gpu can really kick in. For small tensors/models I wouldn't recommend it's use.

# Important Notes

1. Tensorflow uses a computational graph based approach which can be very deceptive. Please track all the variables you create and see to it that your computational graph does not progressively  get bigger each iteration. This can happen if you are unintentionally adding new nodes to the graph. (This is a tensorflow thing not specific to this repo's content).

# If you find this useful, and can make improvements to this regarding speed, and implementation in general, you're free to use this code.

# To Illustrate the gradient-less-ness of this method, I have made a (not so good) numerical optimizer which works (albeit very slowly). This should give you an idea of how to go about implementing something like this for your own optimizer.

# What's next?

I will (hopefully) be adding support for pygmo, so that you can optimize NNs using any optimizer (and there a a lot of them) in pygmo.
