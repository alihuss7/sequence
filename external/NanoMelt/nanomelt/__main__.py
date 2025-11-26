"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
"""

import argparse
import sys

from . import predict
from . import train

USAGE = """

%(prog)s <command> [options]

NanoMelt provides one command:
    - predict: predict the apparent melting temperatures of given nanobody sequences
    - train: train NanoMelt with a new input dataset

see also
%(prog)s <command> predict -h
%(prog)s <command> train -h
for additional help

Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
"""


def main():

    if len(sys.argv) == 1:
        empty_parser = argparse.ArgumentParser(
            description="Semi-supervised ensemble model trained to predict nanobody thermostability",
            usage=USAGE
        )
        empty_parser.print_help(sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Semi-supervised ensemble model trained to predict nanobody thermostability",
    )

    subparser = parser.add_subparsers()

    # PREDICT
    predict_parser = subparser.add_parser("predict",
                                        description="Predict nanobody apparent melting temperatures with NanoMelt",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    predict_parser.add_argument('-i', '--input_filepath_or_seq', help='Filepath to the fasta file .fa to score or directly \
                              a single string sequence', type=str,
                              default='to_score.fa')
    
    predict_parser.add_argument('-o', '--output_savefp', help='Filename of the .csv file to save the predictions in', type=str,
                              default='nanomelt_run.csv')

    predict_parser.add_argument('-align', '--do_align', help='Do the alignment and the cleaning of the given sequences before training. \
                              This step can takes a lot of time if the number of sequences is huge.', action="store_true")
    
    predict_parser.add_argument('-ncpu', '--ncpu', help='If ncpu>1 will parallelise the alignment process', type=int, default=1)

    predict_parser.add_argument('-m', '--maximum_sequences_to_process', help='Internal purpose.', type=int, default=0)

    predict_parser.add_argument('-v', '--verbose', help='Print more details about every step.', action="store_true")

    predict_parser.set_defaults(func=lambda args: predict.run(args))

    # TRAIN
    trainer_parser = subparser.add_parser("train",
                                        description="Train NanoMelt with a new input dataset.",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    trainer_parser.add_argument('-i', '--input_csv', help='Filepath to the .csv file of the nanobody dataset. \
                                It needs to have at least columns with Id, Sequence, Experimental method \
                             and Measured apparent melting temperature (C).', type=str,
                              default='new_nano_thermostability_dataset.csv')
    
    trainer_parser.add_argument('-od', '--output_dir', help='Filepath to the output dir where to save the saved models', type=str,
                              default='nanomelt_retrain')

    trainer_parser.add_argument('-align', '--do_align', help='Do the alignment and the cleaning of the given sequences before training. \
                              This step can takes a lot of time if the number of sequences is huge.', action="store_true")
    
    trainer_parser.add_argument('-ncpu', '--ncpu', help='If ncpu>1 will parallelise the alignment process', type=int, default=1)

    trainer_parser.set_defaults(func=lambda args: train.run(args))


    args = parser.parse_args()
    args.func(args)



if __name__ == "__main__":
    main()
