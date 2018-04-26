from __future__ import absolute_import
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, GRU, Masking, Dropout, Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import LastTimestep
from mimic3models.keras_utils import ExtendMask


class Network(Model):
    
    def __init__(self, dim, batch_norm, dropout, rec_dropout, task,
                target_repl=False, deep_supervision=False, num_classes=1,
                depth=1, input_dim=76, **kwargs):

        print "==> not used params in network class:", kwargs.keys()

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth

        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            return ValueError("Wrong value for task")

        # Input layers and masking
        X = Input(shape=(None, input_dim), name='X')
        inputs = [X]
        mX = Masking()(X)

        if deep_supervision:
            M = Input(shape=(None,), name='M')
            inputs.append(M)

        # Configurations
        is_bidirectional = True
        if deep_supervision:
            is_bidirectional = False

        mX2 = mX

        # Main part of the network
        for i in range(depth - 1):
            num_units = dim
            if is_bidirectional:
                num_units = num_units // 2

            gru = GRU(units=num_units,
                       activation='tanh',
                       return_sequences=True,
                       recurrent_dropout=rec_dropout,
                       dropout=dropout)
            gru2 = GRU(units=num_units,
                       activation='tanh',
                       return_sequences=True,
                       recurrent_dropout=rec_dropout,
                       dropout=dropout)

            if is_bidirectional:
                mX = Bidirectional(gru)(mX)
                mX2 = Bidirectional(gru)(mX2)
            else:
                mX = gru(mX)
                mX2 = gru(mX2)

        # Output module of the network
        return_sequences = (target_repl or deep_supervision)
        L = GRU(units=dim,
                 activation='tanh',
                 return_sequences=return_sequences,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(mX)
        L2 = GRU(units=dim,
                 activation='tanh',
                 return_sequences=return_sequences,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(mX2)

        if dropout > 0:
            L = Dropout(dropout)(L)
            L2 = Dropout(dropout)(L2)

        if target_repl:
            y = TimeDistributed(Dense(num_classes, activation=final_activation),
                                name='seq')(Concatenate()([L, L2]))
            y_last = LastTimestep(name='single')(y)
            outputs = [y_last, y]
        elif deep_supervision:
            y = TimeDistributed(Dense(num_classes, activation=final_activation))(Concatenate()([L, L2]))
            y = ExtendMask()([y, M]) # this way we extend mask of y to M
            outputs = [y]
        else:
            y = Dense(num_classes, activation=final_activation)(Concatenate()([L, L2]))
            outputs = [y]

        return super(Network, self).__init__(inputs=inputs,
                                             outputs=outputs)

    def say_name(self):
        self.network_class_name = "k_gru2"
        return "{}.n{}{}{}{}.dep{}".format(self.network_class_name,
                    self.dim,
                    ".bn" if self.batch_norm else "",
                    ".d{}".format(self.dropout) if self.dropout > 0 else "",
                    ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                    self.depth)