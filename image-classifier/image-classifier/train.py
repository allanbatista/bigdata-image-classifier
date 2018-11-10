#!/usr/bin/env python

import os
import argparse
from datetime import datetime as df

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers

from parameters import initialise_hyper_params


def run(params):
    # Load Image Net
    vgg_conv = vgg16.VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(256, 256, 3))

    # # Dataset
    # train_dir = '/media/thabata/ExtraDrive1/INFNET/Flower_spotter/subset_train'
    # validation_dir = '/media/thabata/ExtraDrive1/INFNET/Flower_spotter/subset_test'

    # nTrain = 723
    # nVal = 239
    #
    # datagen = ImageDataGenerator(rescale=1. / 255)
    #
    # ## train generator
    # train_generator = datagen.flow_from_directory(
    #     train_dir,
    #     target_size=(256, 256),
    #     batch_size=params.batch_size,
    #     class_mode='categorical',
    #     shuffle=True)
    #
    # # load images and generate batch for validation data
    # val_generator = datagen.flow_from_directory(
    #     validation_dir,
    #     target_size=(256, 256),
    #     batch_size=params.batch_size,
    #     class_mode='categorical',
    #     shuffle=True)
    #
    # # data for training
    # train_features = np.zeros(shape=(nTrain, 8, 8, 512))
    # train_labels = np.zeros(shape=(nTrain, 3))
    #
    # i = 0
    # for inputs_batch, labels_batch in train_generator:
    #     features_batch = vgg_conv.predict(inputs_batch)
    #     train_features[i * params.batch_size: (i + 1) * params.batch_size] = features_batch
    #     train_labels[i * params.batch_size: (i + 1) * params.batch_size] = labels_batch
    #     i += 1
    #     if i * params.batch_size >= nTrain:
    #         break
    #
    # train_features = np.reshape(train_features, (nTrain, 8 * 8 * 512))
    #
    # # for validation data
    # val_features = np.zeros(shape=(nVal, 8, 8, 512))
    # val_labels = np.zeros(shape=(nVal, 3))
    #
    # i = 0
    # for inputs_batch, labels_batch in val_generator:
    #     features_batch = vgg_conv.predict(inputs_batch)
    #     val_features[i * params.batch_size: (i + 1) * params.batch_size] = features_batch
    #     val_labels[i * params.batch_size: (i + 1) * params.batch_size] = labels_batch
    #     i += 1
    #     if i * params.batch_size >= nVal:
    #         break
    #
    # val_features = np.reshape(val_features, (nVal, 8 * 8 * 512))
    #
    # # Create a Model
    # model = models.Sequential()
    # model.add(layers.Dropout(0.5))  # minimize overfitting (deactivates half of the neurons)
    # model.add(layers.Dense(256, activation='relu', input_dim=8 * 8 * 512))
    # model.add(layers.Dense(3, activation='softmax'))
    #
    # model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
    #               loss='categorical_crossentropy',
    #               metrics=['acc'])
    #
    # metrics = model.fit(train_features,
    #                     train_labels,
    #                     epochs=20,
    #                     batch_size=params.batch_size,
    #                     validation_data=(val_features, val_labels))
    #
    # ## check performance
    # fnames = val_generator.filenames
    # ground_truth = val_generator.classes
    # label2index = val_generator.class_indices
    #
    # # Getting the mapping from class index to class label
    # idx2label = dict((v, k) for k, v in label2index.items())
    #
    # predictions = model.predict_classes(val_features)
    # prob = model.predict(val_features)
    #
    # errors = np.where(predictions != ground_truth)[0]
    # print("No of errors = {}/{}".format(len(errors), nVal))


def main():

    print('')
    print('Hyper-parameters:')
    print(HYPER_PARAMS)
    print('')

    print("$ tensorboard --port 8080 --logdir {}".format(HYPER_PARAMS.tensorboards_path))

    # Set python level verbosity
    tf.logging.set_verbosity(HYPER_PARAMS.verbosity)

    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[HYPER_PARAMS.verbosity] / 10)

    # Run the train and evaluate experiment
    time_start = df.utcnow()
    print("")
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    run(HYPER_PARAMS)

    time_end = df.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print("")


args_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialise_hyper_params(args_parser)

if __name__ == '__main__':
    main()
