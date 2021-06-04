import tensorflow as tf

phx = tf.placeholder('float', [None, None])
phy = tf.placeholder('float')
phhl = tf.placeholder('float') #hinge_loss


hidden_layer_1 = 512
hidden_layer_2 = 32
output_layer = 1
batch_size = 32


def define_network(X):
    hidden_1 = {'weights':tf.Variable(tf.random_normal([4096, hidden_layer_1])), 'biases':tf.Variable(tf.random_normal([hidden_layer_1]))}

    hidden_2 = {'weights':tf.Variable(tf.random_normal([hidden_layer_1, hidden_layer_2])), 'biases':tf.Variable(tf.random_normal([hidden_layer_2]))}

    output = {'weights':tf.Variable(tf.random_normal([hidden_layer_2, output_layer])), 'biases':tf.Variable(tf.random_normal([output_layer])),}


    layer_1 = tf.add(tf.matmul(X,hidden_1['weights']), hidden_1['biases'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.layers.dropout(layer_1, 0.6)
    
    
    layer_2 = tf.add(tf.matmul(layer_1,hidden_2['weights']), hidden_2['biases'])
    layer_2  = tf.layers.dropout(layer_2, 0.6)
    
    
    layer_output = tf.matmul(layer_2,output['weights']) + output['biases']
    layer_output = tf.nn.sigmoid(layer_output)
    

    return layer_output , hidden_1['weights'] , hidden_2['weights'] , output['weights']


