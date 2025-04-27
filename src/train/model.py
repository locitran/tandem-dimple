import tensorflow as tf

class DNN(tf.keras.Model):
    """
    names = ['input', 'hidden_1', 'hidden_2', ..., 'output']
    nodes = [33, 33, 33, 33, 33, 33, 10, 2]
    """

    def __init__(self, names, nodes, dropout=None, output_bias=None):
        super(DNN, self).__init__()
        assert len(names) == len(nodes), "Length of names and nodes must be equal"
        self.names = names
        self.nodes = nodes

        self.hidden_layers = []

        for i in range(len(names)):
            if i == 0:
                # Input layer: define input shape
                layer = tf.keras.layers.Dense(nodes[i], kernel_initializer='glorot_uniform',name=names[i])
            elif i == len(names) - 1:
                # Output layer
                bias_initializer = tf.keras.initializers.Constant(output_bias) if output_bias is not None else 'zeros'
                layer = tf.keras.layers.Dense(nodes[i], kernel_initializer='glorot_uniform', bias_initializer=bias_initializer, name=names[i])
            else:
                # Hidden layers
                layer = tf.keras.layers.Dense(nodes[i],kernel_initializer='glorot_uniform',name=names[i])
            self.hidden_layers.append(layer)

        self.gelu = tf.keras.layers.Activation('gelu')
        self.softmax = tf.keras.layers.Activation('softmax')
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout is not None else None

    def call(self, x, training=False):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if i < len(self.hidden_layers) - 1:
                x = self.gelu(x)
                if self.dropout is not None:
                    x = self.dropout(x, training=training)
            else:
                x = self.softmax(x)
        return x
    

# Build model 
# def build_model(cfg, output_bias=None):
#     """Build a neural network model based on the configuration file
#     Args:
#         cfg: dict
#             Configuration file
#         output_bias: np.array
#             Initial bias for the output layer

#     Returns:
#         model: tf.keras.Model
#             Neural network model
#     """
#     if output_bias is not None:
#         output_bias = tf.keras.initializers.Constant(output_bias)
#     # X = tf.keras.Input(shape=cfg['model']['input']['n_neurons'])
#     input_shape = (cfg.model.input.n_neurons,)  # Define input shape as a tuple
#     X = tf.keras.Input(shape=input_shape)
#     Y = X
#     for hid_name, hid_config in cfg['model']['hidden'].items():
#         Y = tf.keras.layers.Dense(units=hid_config['n_neurons'], activation=hid_config['activation'], kernel_initializer=hid_config['initializer'],
#                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=hid_config['l1'], l2=hid_config['l2']),
#                 name=hid_name)(Y)
#         if hid_config['batch_norm']:
#             Y = tf.keras.layers.BatchNormalization()(Y)
#         Y = tf.keras.layers.Dropout(hid_config['dropout_rate'])(Y)
#     Y = tf.keras.layers.Dense(units=cfg['model']['output']['n_neurons'], activation=cfg['model']['output']['activation'], name='Output')(Y)
#     model = tf.keras.Model(inputs=X, outputs=Y, name="IMPROVE")
#     return model