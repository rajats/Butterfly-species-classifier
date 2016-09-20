# Butterfly-species-classifier
Butterfly species classifier was implemented using Theano and Lasagne. Convolutional Neural Network was used to classify butterfly species. 
Lasagne layers was used to construct Convolutional Neural Network. 
Two different methods were used to implement the classifier: 

# Method 1 
**Using pre-trained weights of VGG16 net** *[Simonyan and Zisserman, 2014]*   

In this method a pre-trained [VGG16 network](https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py) was taken from Lasagne model zoo, since VGG16 network was trained for ImageNet Large Scale Visual Recognition Challenge (ILSVRC) which has 1000 categories, a output classifier for butterfly species was connected to the last fully connected layer of the VGG16 network, so that we can get probabilities/score for 10 categories(butterfly species). After that network was trained for 20 epochs. This model gave best validation accuracy of 98% and test accuracy of 99%.

# Method 2  

In this method Convolutional Neural Network was created with architecture of type Conv-ReLu-Conv-ReLu-MaxPool-Conv-ReLu-Conv-ReLu-MaxPool-Affine-Dropout-Affine-dropout-Affine-Softmax, where first two convolution layers had 64 filters and next two convolution layers had 128 filters, both affine layers had 256 hidden neurons with dropout of 50%.  

This network was trained for 60 epochs. There was noticeable overfitting after training. This model gave best validation accuracy of 72% and test accuracy of 81.25%.  
Network was trained with random values of hyperparameters like regularization parameter, learning rate etc. Performance can be improved by validation of hyperparameters.  

# Performance 
Clearly network trained with pre-trained weights gave far more better performance than method2.  

![alt tag](https://raw.githubusercontent.com/rajats/Butterfly-species-classifier/master/result1.PNG)  

![alt tag](https://raw.githubusercontent.com/rajats/Butterfly-species-classifier/master/result2.PNG)



