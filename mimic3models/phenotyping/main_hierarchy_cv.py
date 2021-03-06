import numpy as np
import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import imp
import re

from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReaderHCV

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from sklearn.model_selection import KFold

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

listfile_path = '../../data/phenotyping/cv_listfile.csv'
with open(listfile_path, "r") as lfile:
    listfile_data = lfile.readlines()
listfile_data = np.array(listfile_data[1:])
kf = KFold(n_splits = 5)
for train_index, test_index in kf.split(listfile_data):

    # Build readers, discretizers, normalizers
    train_reader = PhenotypingReaderHCV(dataset_dir='../../data/phenotyping/train/',
                                    listfile='../../data/phenotyping/val_listfile.csv', listfile_data = np.ndarray.tolist(listfile_data[train_index]))

    val_reader = PhenotypingReaderHCV(dataset_dir='../../data/phenotyping/train/',
                                listfile='../../data/phenotyping/val_listfile.csv', listfile_data = np.ndarray.tolist(listfile_data[test_index]))

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

    # Load model weights
    n_trained_chunks = 0
    if args.load_state != "":
        model.load_weights(args.load_state)
        n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


    # Build data generators
    train_data_gen = utils.BatchGen(train_reader, discretizer,
                                    normalizer, args.batch_size,
                                    args.small_part, target_repl, shuffle=True)
    val_data_gen = utils.BatchGen(val_reader, discretizer,
                                normalizer, args.batch_size,
                                args.small_part, target_repl, shuffle=False)

    if args.mode == 'train':
        # Prepare training
        path = 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state'

        metrics_callback = keras_utils.PhenotypingMetricsH(train_data_gen=train_data_gen,
                                                        val_data_gen=val_data_gen,
                                                        batch_size=args.batch_size,
                                                        verbose=args.verbose)
        # make sure save directory exists
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        saver = ModelCheckpoint(path, monitor='val_loss', verbose=1, period=args.save_every, save_best_only=True)

        if not os.path.exists('keras_logs'):
            os.makedirs('keras_logs')
        csv_logger = CSVLogger(os.path.join('keras_logs', model.final_name + '.csv'),
                            append=True, separator=';')

        print "==> training"
        model.fit_generator(generator=train_data_gen,
                            steps_per_epoch=train_data_gen.steps,
                            validation_data=val_data_gen,
                            validation_steps=val_data_gen.steps,
                            epochs=n_trained_chunks + args.epochs,
                            initial_epoch=n_trained_chunks,
                            callbacks=[metrics_callback, saver, csv_logger],
                            verbose=args.verbose)

    elif args.mode == 'test':

        # ensure that the code uses test_reader
        del train_reader
        del val_reader
        del train_data_gen
        del val_data_gen

        test_reader = PhenotypingReaderHCV(dataset_dir='../../data/phenotyping/test/',
                                        listfile='../../data/phenotyping/test_listfile.csv')

        test_data_gen = utils.BatchGen(test_reader, discretizer,
                                    normalizer, args.batch_size,
                                    args.small_part, target_repl,
                                    shuffle=False, return_names=True)

        names = []
        ts = []
        labels = []
        predictions = []
        for i in range(test_data_gen.steps):
            print "\rpredicting {} / {}".format(i, test_data_gen.steps),
            ret = next(test_data_gen)
            x = ret["data"][0]
            y = ret["data"][1]
            cur_names = ret["names"]
            cur_ts = ret["ts"]
            x = np.array(x)
            pred = model.predict_on_batch(x)
            predictions += list(pred)
            labels += list(y)
            names += list(cur_names)
            ts += list(cur_ts)

        metrics.print_metrics_multilabel(labels, predictions)
        path = os.path.join("test_predictions", os.path.basename(args.load_state)) + ".csv"
        utils.save_results(names, ts, predictions, labels, path)

    else:
        raise ValueError("Wrong value for args.mode")