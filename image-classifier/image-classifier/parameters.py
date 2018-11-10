#!/usr/bin/env python

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


def initialise_hyper_params(args_parser):
    args_parser.add_argument(
        '--job-dir',
        help='jobs staging path',
        required=True
    )
    args_parser.add_argument(
        '--batch-size',
        help='batch size',
        default=100,
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
        default="trainset/"
    )
    args_parser.add_argument(
        '--testset-path',
        help='GCS pattern or local paths to evaluation data',
        default="testset/"
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

    return Arguments(args_parser.parse_args())