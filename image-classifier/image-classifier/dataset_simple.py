#!/usr/bin/env python

import tensorflow as tf
from google.cloud import storage

gs = storage.Client()
bucket = gs.bucket('bigdata-allanbatista-com-br')


def _parse_example(serialized_example):
    feature_set = {
        'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'features': tf.FixedLenFeature([], tf.string)
    }

    parsed = tf.parse_single_example(serialized_example, features=feature_set)

    reshaped = tf.reshape(tf.decode_raw(parsed['features'], tf.float32), (256, 256, 3))

    return reshaped, parsed['label']


def create_dataset(files, batch_size=16, epochs=1):
    print("build_tfrecord: files_count: {} / batch_size: {} / epochs: {}".format(len(files), batch_size, epochs))

    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(_parse_example)
    ds = ds.batch(batch_size)
    ds = ds.repeat(epochs)
    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)

    return ds


def blobs_names_with_bucket(bucket_name, pattern):
    return ["gs://%s/%s" % (bucket_name, i.name) for i in bucket.list_blobs(prefix=pattern)]


def create_dataset_form_pattern(bucket_name, pattern, batch_size=16, epochs=1):
    return create_dataset(blobs_names_with_bucket(bucket_name, pattern), batch_size, epochs)



