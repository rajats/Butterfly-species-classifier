# Butterfly-species-classifier
Butterfly species classifier was implemented using Theano and Lasagne. Convolutional Neural Network was used to classify butterfly species. 
Lasagne layers was used to construct Convolutional Neural Network. 
Two different methods were used to implement the classifier: 

# Method 1 
**Using pre-trained weights of VGG16 net *[Simonyan and Zisserman, 2014]* ** 
In this method a pre-trained [VGG16 network](https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py) was taken from Lasagne model zoo, since VGG16 network was trained for ImageNet Large Scale Visual Recognition Challenge (ILSVRC) which has 1000 categories, a output classifier for butterfly species was connected to the last fully connected layer of the VGG16 network, so that we can get probabilities/score for 10 categories(butterfly species). After that network was trained for 20 epochs. This model gave best validation accuracy of 98% and test accuracy of 96.6%.
