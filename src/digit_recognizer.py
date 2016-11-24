import csv
import numpy as np
import argparse

import convolutional_neural_network as cnn
import feed_forward_net as ffn
import feed_forward_net_optimized as ffno


# Matplotlib - A Library for plots
import matplotlib.pyplot as plt

def load_data(filename, split=0.75):
    fp = open(filename, 'rb')
    data = csv.reader(fp, delimiter=',')
    data_x = []
    data_y = []
    next(data)
    for row in data:
        data_y.append(int(row[0]))
        cols = np.zeros(784)
        for num in range(784):
            cols[num] = float(row[num+1])/255.0
        data_x.append(cols)
    fp.close()
    train_len = int(len(data_x)*split)
    training_data = np.asarray(data_x[:train_len]), np.asarray(data_y[:train_len])
    validation_data = np.asarray(data_x[train_len:]), np.asarray(data_y[train_len:])
    return training_data, validation_data

def vectorized(d):
    vec = np.zeros((10, 1))
    vec[d] = 1.0
    return vec


def load_test_data(filename):
    fp = open(filename, 'rb')
    data = csv.reader(fp, delimiter=',')
    data_x = []
    data_y = []
    index = 0
    next(data)
    for row in data:
        cols = []
        for col in row:
            cols.append(float(col)/255.0)
        data_x.append(cols)
    fp.close()
    return np.asarray(data_x)


def load_data_wrapper(train_file, test_file, split=0.75):
    train_data, validation_data = load_data(train_file, split)
    train_x, train_y = train_data
    training_data_inputs = [ np.reshape(x, (784,1)) for x in train_x]
    training_data_results = [vectorized(y) for y in train_y]
    training_data = zip(training_data_inputs, training_data_results)
    validation_x, validation_y = validation_data
    valdiation_inputs = [np.reshape(x, (784,1)) for x in validation_x]
    validation_data = zip(valdiation_inputs, validation_y)
    tst_dt = load_test_data(test_file)
    test_data = [ np.reshape(x, (784, 1)) for x in tst_dt]
    return (training_data, validation_data, test_data)


def write_output(filename, test_labels):
    fp = open(filename, 'wb')
    fp.write('ImageId,Label\n')
    imageId = 1
    for label in test_labels:
        fp.write('{0},{1}\n'.format(imageId,label))
        imageId += 1
    fp.close()

def plot_accuracies(x_labels, y_values, title, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_labels,
            y_values,
            color = '#2A6EA6')
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()

""" Main function to be run. """
def main(**kawrgs):
    
    print 'ffn  - Basic Feed Forward Neural Network with Sigmoid Activations and Quadratic Cost\n' +\
          'ffno - Feed Forward Neural Network with Sigmoid Activations and Cross Entropy Cost and Better Weight Initialization\n' +\
          'cnn - Network with 2 Conv Layers with Rectified Linear Units followed by Max Pooling, 1 Fully Connected layer and a Soft Max layer\n'
    _dir = '../output/'
    # net_type = raw_input('Enter type of network you want to use :')
    # output_file = raw_input("Enter name of the output file :")

    net_type = 'cnn'
    output_file = 'output.csv'
    train_file = '../data/train.csv'
    test_file = '../data/test.csv'

    if kawrgs.get('s'):
        split = kawrgs.get('s')
    else:
        split = 0.75
    training_data, validation_data, test_data = load_data_wrapper(train_file, test_file, split=split)
    # Default Network
    
    test_labels = []
    if net_type == "ffno":
        net = ffno.Network([784, 30, 10], cost = ffno.CrossEntropyCost)
        print 'Training Feed Forward Net with Cross Entropy Cost and '
        net.StochasticGradientDescent(training_data, epochs, mini_batch_size, eta, validation_data=validation_data, lmda=5.0)
        test_labels = net.predict(test_data)
    elif net_type == "cnn":
        training_data, validation_data = load_data(train_file, split=split)
        shared_training_data = cnn.shared(training_data)
        shared_validation_data = cnn.shared(validation_data)
        epochs = 60
        mini_batch_size = 10
        eta = 0.1
        lmda=0.1
        cnet = cnn.Network([cnn.ConvPoolLayer(image_shape=(mini_batch_size, 1 , 28, 28),
                                     filter_shape=(20, 1, 5, 5),
                                     poolsize=(2,2), activation_fn = cnn.ReLu),
                       cnn.FullyConnectedLayer(n_in=20*12*12, n_out=100),
                       cnn.SoftMaxLayer(n_in=100, n_out=10)], mini_batch_size)
        print 'Training CNN using RELU as activation function for Conv Layers with regularization'
        bva = cnet.SGD(shared_training_data, epochs, mini_batch_size, eta, lmda=lmda)
        print 'Best Validation Accuracy with Conv Nets using ReLU :{0}'.format(bva)
        test_labels = cnet.predict(load_test_data(test_file))
    else:
        net = ffn.Network([784, 30, 10])
        print 'Training a basic feed forward net with 1 hidden layer for',epochs,'epochs'
        net.StochasticGradientDescent(training_data, epochs, mini_batch_size, eta, validation_data=shared_validation_data)
        test_labels = net.predict(test_data, mini_batch_size)
    

    print 'Training Done'
    write_output(_dir+output_file, test_labels)
    print 'Test labels are written to file:',_dir+output_file
    lmbdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    accuracies = []
    for lmbda in lmbdas:
        bva = cnet.SGD(shared_training_data, epochs, mini_batch_size, eta, validation_data=shared_validation_data, lmda=lmda)
        accuracies.append(lmbdas)
    plot_accuracies(lmbdas, accuracies, 'Lambdas', 'Best Accuracy','Validation Accuracies vs Lambdas')

if __name__ == "__main__":
    import os
    import subprocess
    import sys

    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32'
    # subprocess.check_call(['sqsub', '-np', sys.argv[1], '/path/to/executable'],
    #                       env=dict(os.environ, SQSUB_VAR="visible in this subprocess"))
    
    main()
