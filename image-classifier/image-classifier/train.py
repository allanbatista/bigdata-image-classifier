#!/usr/bin/env python

import os
import json
import argparse
from datetime import datetime as df

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

import output
import dataset
from parameters import initialise_hyper_params
from tensorflow.python.lib.io import file_io


def run(params):
    print(params.metadata_filename)
    with file_io.FileIO(params.metadata_filename, 'r') as f:
        metadata = json.loads(f.read())

    trainset = dataset.create_dataset_form_pattern(params.bucket_name,
                                                   params.trainset_path,
                                                   batch_size=params.batch_size)

    testset  = dataset.create_dataset_form_pattern(params.bucket_name,
                                                   params.testset_path,
                                                   batch_size=params.batch_size)

    # # Create a Model
    model = models.Sequential()

    first_data = True
    print(params.layers)
    for layer_data in params.layers:
        data = layer_data['params']

        if first_data:
            data['input_dim'] = metadata['input_dimention']
            first_data = False

        model.add(getattr(layers, layer_data['class_name'])(**data))

    model.add(layers.Dense(metadata['classes_count'], activation='softmax'))


    compile_data = {
        'optimizer': getattr(optimizers, params.optimizer)(**params.optimizer_params),
        'loss': params.loss,
        'metrics': params.metrics
    }
    print(compile_data)
    model.compile(**compile_data)

    model.fit(trainset,
                epochs=params.epochs,
                steps_per_epoch=metadata['train_samples_count'] // params.batch_size,
                validation_data=testset,
                validation_steps=metadata['test_samples_count'] // params.batch_size,
                callbacks=[
                    tf.keras.callbacks.TensorBoard(log_dir=params.tensorboards_path,
                                                   batch_size=params.batch_size,
                                                   write_graph=True,
                                                   write_grads=True,
                                                   write_images=True),
                    tf.keras.callbacks.EarlyStopping(monitor=params.early_stopping_monitor,
                                                     min_delta=params.early_stopping_delta,
                                                     verbose=params.early_stopping_verbose,
                                                     patience=params.early_stopping_patience)
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
