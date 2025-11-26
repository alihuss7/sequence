#!/usr/bin/env python
import argparse
import sys,os
#sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../../abnativ_git/abnativ/")) # only needed in webserver, comment out otherwise
from nanomelt import predict

"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
"""


USAGE = """

%(prog)s <command> [options]

NanoMelt provides one command:
    - predict: predict the apparent melting temperatures of given nanobody sequences

see also
%(prog)s <command> predict -h
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

    predict_parser.add_argument('-v', '--verbose', help='Print more details about every step', action="store_true")

    predict_parser.add_argument('-maxNseq', '--maximum_sequences_to_process', help='If set to 0 this will be ingored. Only useful on webserver, limits the maximum number of sequences that can be processed when giving a fasta file as input, raises error if above.', type=int, default=0)

    predict_parser.set_defaults(func=lambda args: predict.run(args))

   
    args = parser.parse_args()
    args.func(args)



if __name__ == "__main__":
    main()
