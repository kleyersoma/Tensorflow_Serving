# Tensorflow Serving

Building Robust Production-Ready Deep Learning Vision Models in Minutes, with Tensorflow Serving models can be served into production.

## What is TensorFlow Serving?

Is an API to use or apply with a model for inference after it has been trained, it involves having a 
server-client architecture and serving or exposing our trained models for inference.

In this reposiory an image classifier is developed and certain specific _access patterns_ will be applied for the model.
A brief overview of what these _access patterns_ are:

* On the client side there is an input image which need to be classified. This image needs to be converted to an specific encoded format.
* The image must be wrapped in a specific JSON payload with headers.
* Then the image is sent to a web services/API which should typically be hosted on a server. 
* The API call will invoke the pre-trained model to make the prediction and serve the inference result as a JSON response from the 
  server to the client.
 * The client gets its predictions and hopefully, it will be useful.
 
