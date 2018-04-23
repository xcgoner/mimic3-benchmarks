from __future__ import absolute_import
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, GRU, Masking, Dropout, Add, Multiply, Concatenate, Lambda
from keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import LastTimestep
from mimic3models.keras_utils import ExtendMask
from mimic3models.phenotyping import utils


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

            if is_bidirectional:
                mX = Bidirectional(gru)(mX)
            else:
                mX = gru(mX)

        # Output module of the network
        return_sequences = (target_repl or deep_supervision)
        L_lv1 = GRU(units=dim,
                 activation='tanh',
                 return_sequences=True,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(mX)

        L = L_lv1

        label_struct = utils.read_hierarchical_labels('../../data/phenotyping/label_list.txt', '../../data/phenotyping/label_struct.json')
        # only support 2 levels
        num_superclass = len(label_struct.keys())

        output_lv1 = Lambda(lambda x: x[:,-1,:])(L)
        # if dropout > 0:
        #     output_lv1 = Dropout(dropout)(output_lv1)
        output_lv1 = Dense(num_superclass, activation=final_activation)(output_lv1)

        L_lv2_gru = GRU(units=dim,
                 activation='tanh',
                 return_sequences=return_sequences,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(L_lv1)
        
        # if dropout > 0:
        #     L_lv2_gru = Dropout(dropout)(L_lv2_gru)

        L_lv2_2 = GRU(units=dim,
                 activation='tanh',
                 return_sequences=return_sequences,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(mX)

        output_lv2 = Dense(25, activation=final_activation)(Concatenate()([L_lv2_gru, L_lv2_2]))

        y = Concatenate()([output_lv2, output_lv1])
        outputs = [y]

        return super(Network, self).__init__(inputs=inputs,
                                             outputs=outputs)

    def say_name(self):
        self.network_class_name = "k_hgru_1"
        return "{}.n{}{}{}{}.dep{}".format(self.network_class_name,
                    self.dim,
                    ".bn" if self.batch_norm else "",
                    ".d{}".format(self.dropout) if self.dropout > 0 else "",
                    ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                    self.depth)