#!/usr/bin/env python

import os
import json
import argparse
from datetime import datetime as df

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16

import output
import dataset
from parameters import initialise_hyper_params
from tensorflow.python.lib.io import file_io


def run(params):
    with file_io.FileIO(params.metadata_filename, 'r') as f:
        metadata = json.loads(f.read())

    trainset = dataset.create_dataset_form_pattern(params.bucket_name,
                                                   params.trainset_path,
                                                   batch_size=params.batch_size)

    testset  = dataset.create_dataset_form_pattern(params.bucket_name,
                                                   params.testset_path,
                                                   batch_size=params.batch_size)

    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Create a Model
    model = models.Sequential()
    model.add(vgg_conv)
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(10, activation='relu', input_dim=metadata['input_dimention']))
    model.add(layers.Dense(metadata['classes_count'], activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                  loss=params.loss,
                  metrics=['acc', 'mse'])

    model.fit(trainset,
                epochs=params.epochs,
                steps_per_epoch= 100 // 16, #metadata['train_samples_count'] // params.batch_size,
                # validation_data=testset,
                # validation_steps=metadata['test_samples_count'] // params.batch_size,
                callbacks=[
                    tf.keras.callbacks.TensorBoard(log_dir=params.tensorboards_path,
                                                   batch_size= 100 // 16, #params.batch_size,
                                                   write_graph=True,
                                                   write_grads=True,
                                                   write_images=True),
                    tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, verbose=1, patience=2)
                ])

    print("Saving Model")
    output.save(model, params)


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
