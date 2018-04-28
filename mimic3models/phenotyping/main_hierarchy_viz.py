import numpy as np
import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import imp
import re

from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReaderH

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
args = parser.parse_args()
print args

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = PhenotypingReaderH(dataset_dir='../../data/phenotyping/train/',
                                 listfile='../../data/phenotyping/train_listfile.csv')

val_reader = PhenotypingReaderH(dataset_dir='../../data/phenotyping/train/',
                               listfile='../../data/phenotyping/val_listfile.csv')

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here onlycont vs all
normalizer.load_params('ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(args.timestep))

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ph'
args_dict['num_classes'] = 27
args_dict['target_repl'] = target_repl

# Build the model
print "==> using model {}".format(args.network)
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + model.say_name() + suffix
print "==> model.final_name:", model.final_name


# Compile the model
print "==> compiling the model"
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
    loss = 'binary_crossentropy'
    loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model.summary()

plot_model(model, to_file='../../data/model.png')