import numpy as np
import theano
import theano.tensor as T
import lasagne


def gen_batch(X, y, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('int32')
        

def train(net, X_train, y_train, X_val, y_val, learning_rate=1e-4, learning_rate_decay=0.95,
          decay_after_epochs=4, reg=0.001, batch_size=50, num_epochs=20):
    best_val_acc = 0.0
    best_model = None
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    X_b = T.tensor4()
    y_b = T.ivector()
    output = lasagne.layers.get_output(net['scores'], X_b)
    pred = output.argmax(-1)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(output, y_b))
    reg_loss = lasagne.regularization.regularize_network_params(net['scores'], lasagne.regularization.l2)
    loss = loss + reg * reg_loss
    acc = T.mean(T.eq(pred, y_b))
    params = lasagne.layers.get_all_params(net['scores'],  trainable=True)
    grad = T.grad(loss, params)
    lr = theano.shared(np.float32(learning_rate))
    updates = lasagne.updates.adam(grad, params, learning_rate=lr)
    f_train = theano.function([X_b, y_b], [loss, acc], updates=updates)
    f_val = theano.function([X_b, y_b], [loss, acc])
    f_predict = theano.function([X_b], pred)
    num_batches = len(X_train) // batch_size
    num_val_batches = len(X_val) // batch_size
    train_batches = gen_batch(X_train, y_train, batch_size)
    val_batches = gen_batch(X_val, y_val, batch_size)
    print "training"
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        for _ in range(num_batches):
            X, y = next(train_batches)
            loss, acc = f_train(X, y)
            loss_history.append(loss)
            train_loss += loss
            train_acc += acc
        train_loss /= num_batches
        train_acc /= num_batches
        train_acc_history.append(train_acc)

        val_loss = 0
        val_acc = 0
        for _ in range(num_val_batches):
            X, y = next(val_batches)
            loss, acc = f_val(X, y)
            val_loss += loss
            val_acc += acc
        val_loss /= num_val_batches
        val_acc /= num_val_batches
        val_acc_history.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = lasagne.layers.get_all_param_values(net['scores'])
        
        print 'epoch %d / %d : training loss: %f, training accuracy: %.3f, validation loss: %f, validation accuracy: %.3f'\
            % (epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc)
            
        if (epoch + 1) % decay_after_epochs == 0:
            lr.set_value(np.float32(lr.get_value() * learning_rate_decay))

    return best_model, loss_history, train_acc_history, val_acc_history, f_predict