#!/usr/bin/env python

import os
import argparse
from datetime import datetime as df
import pdb

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

import dataset
from parameters import initialise_hyper_params


def run(params):
    trainset_path = params.trainset_path.replace("gs://bigdata-allanbatista-com-br/", "")
    trainset = dataset.create_dataset_form_pattern(params.bucket_name, trainset_path, batch_size=10)
    # testset  = dataset.create_dataset_form_pattern(params.bucket_name, params.testset_path)

    # # Create a Model
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=256 * 256 * 3))
    model.add(layers.Dropout(0.5))  # minimize overfitting (deactivates half of the neurons)
    model.add(layers.Dense(17, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    metrics = model.fit(trainset,
                        epochs=20,
                        steps_per_epoch=753 // 20)


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
