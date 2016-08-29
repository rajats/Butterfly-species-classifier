import lasagne

def create_cnn(input_shape=(None, 3, 224, 224), num_filters=[64, 128], filter_size=3,
              hidden_dims=[256, 256], num_classes=10):
    net = {}
    net['input'] = lasagne.layers.InputLayer(shape=input_shape)
    
    # conv_relu_x2_pool
    conv_layer_idx = 1
    pool_layer_idx = 1
    for num_filter in num_filters:
        if conv_layer_idx == 1:
            net['conv%s' %(conv_layer_idx)] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['input'], 
                num_filters=num_filter, filter_size=(filter_size, filter_size),
                pad='same', nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(gain='relu')))
        else:    
            net['conv%s' %(conv_layer_idx)] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
                net['pool%s' %(pool_layer_idx-1)], num_filters=num_filter, filter_size=(filter_size, filter_size),
                pad='same', nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(gain='relu')))
        conv_layer_idx += 1
        net['conv%s' %(conv_layer_idx)] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            net['conv%s' %(conv_layer_idx-1)], num_filters=num_filter, filter_size=(filter_size, filter_size),
            pad='same', nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(gain='relu')))
        conv_layer_idx += 1
        net['pool%s' %(pool_layer_idx)] = lasagne.layers.MaxPool2DLayer(net['conv%s' %(conv_layer_idx-1)], pool_size=(2, 2))
        pool_layer_idx += 1
        
    # affine_relu
    affine_layer_idx = 1
    for hidden_dim in hidden_dims:
        if affine_layer_idx == 1:
            net['affine%s' %(affine_layer_idx)] = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
                lasagne.layers.dropout(net['pool%s' %(pool_layer_idx-1)], p=0.5),
                num_units=hidden_dim, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(gain='relu')))
        else:
             net['affine%s' %(affine_layer_idx)] = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
                lasagne.layers.dropout(net['affine%s' %(affine_layer_idx-1)] , p=0.5),
                num_units=hidden_dim, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(gain='relu')))
        affine_layer_idx += 1
        
    # scores
    net['scores'] = lasagne.layers.DenseLayer(net['affine%s' %(affine_layer_idx-1)], 
        num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return net