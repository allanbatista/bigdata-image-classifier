#!/usr/bin/env python

import json
from datetime import datetime as dt


class Arguments():
    def __init__(self, params):
        self.params = params

    @property
    def batch_size(self):
        return self.params.batch_size

    @property
    def epochs(self):
        return self.params.epochs

    @property
    def base_path(self):
        if self.params.base_path[-1] is "/":
            return self.params.base_path
        else:
            return self.params.base_path + "/"

    @property
    def current_date(self):
        return self.params.current_date

    @property
    def metadata_filename(self):
        return self.base_path + self.params.metadata_filename

    @property
    def verbosity(self):
        return self.params.verbosity

    @property
    def tensorboards_path(self):
        return "{}train/{}/tensorboards".format(self.base_path, self.current_date)

    @property
    def model_filename(self):
        return "{}train/{}/model.hdf5".format(self.base_path, self.current_date)

    @property
    def job_dir(self):
        return self.params.job_dir

    @property
    def trainset_path(self):
        return (self.base_path + self.params.trainset_path).replace('gs://bigdata-allanbatista-com-br/', '')

    @property
    def testset_path(self):
        return (self.base_path + self.params.testset_path).replace('gs://bigdata-allanbatista-com-br/', '')

    @property
    def bucket_name(self):
        return self.params.bucket_name

    @property
    def loss(self):
        return self.params.loss

    @property
    def early_stopping_monitor(self):
        return self.params.early_stopping_monitor

    @property
    def early_stopping_delta(self):
        return self.params.early_stopping_delta

    @property
    def early_stopping_verbose(self):
        return self.params.early_stopping_verbose

    @property
    def early_stopping_patience(self):
        return self.params.early_stopping_patience

    @property
    def optimizer(self):
        return self.params.optimizer

    @property
    def optimizer_params(self):
        return json.loads(self.params.optimizer_params)

    @property
    def layers(self):
        return json.loads(self.params.layers)

    @property
    def metrics(self):
        return self.params.metrics.split(",")


def initialise_hyper_params(args_parser):
    args_parser.add_argument(
        '--job-dir',
        help='jobs staging path',
        required=True
    )
    args_parser.add_argument(
        '--bucket-name',
        help='google cloud storage bucket name',
        required=True
    )
    args_parser.add_argument(
        '--batch-size',
        help='batch size',
        default=16,
        type=int
    )
    args_parser.add_argument(
        '--epochs',
        help='epochs',
        default=1,
        type=int
    )
    args_parser.add_argument(
        '--base-path',
        help='GCS Base Path. ex.: gs://bitservices-bigdata/skyhub/minos/20185006-195016/tfidf/',
        required=True
    )
    args_parser.add_argument(
        '--current-date',
        help='Current Date ex.: 20181108_130726',
        default=dt.now().strftime("%Y%m%d_%H%M%S")
    )
    # Data files arguments
    args_parser.add_argument(
        '--trainset-path',
        help='GCS pattern or local paths to training data',
        default='trainset/'
    )
    args_parser.add_argument(
        '--testset-path',
        help='GCS pattern or local paths to evaluation data',
        default='testset'
    )
    args_parser.add_argument(
        '--metadata-filename',
        help='GCS or local paths to metadata data',
        default='metadata.json'
    )
    # Argument to turn on all logging
    args_parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='DEBUG',
    )

    args_parser.add_argument(
        '--loss',
        default='categorical_crossentropy',
        help='To see loss function compatible see https://keras.io/losses/'
    )

    args_parser.add_argument(
        '--early-stopping-monitor',
        default='loss'
    )

    args_parser.add_argument(
        '--early-stopping-delta',
        default=0.01,
        type=float
    )

    args_parser.add_argument(
        '--early-stopping-verbose',
        default=1,
        type=int
    )

    args_parser.add_argument(
        '--early-stopping-patience',
        default=2,
        type=int
    )

    args_parser.add_argument(
        '--optimizer',
        default='RMSprop',
        help='See more about optimizers in https://keras.io/optimizers/'
    )

    args_parser.add_argument(
        '--optimizer_params',
        default='{"lr":0.001}',
        help='JSON DATA, See more about optimizers in https://keras.io/optimizers/'
    )

    args_parser.add_argument(
        '--layers',
        default='[{"class_name": "Dense", "params": {"units": 10, "activation":"relu"}}, {"class_name": "Dropout", "params":{"rate":0.1}}]',
        help='JSON DATA.'
    )

    args_parser.add_argument(
        '--metrics',
        default='acc,mse',
        help='ex.: acc,mse'
    )

    return Arguments(args_parser.parse_args())