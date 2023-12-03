import argparse
from .dataset import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict clips using BERT models')
    subparsers = parser.add_subparsers(required=True, title='subcommands', dest='command')

    subparser = subparsers.add_parser('dataset', help='create CSV dataset from .markermap files')
    subparser.add_argument('--input', '-i', required=True, help='the folder where to search *.markermap files')
    subparser.add_argument('--output', '-o', required=True, help='output JSON path')
    

    subparser = subparsers.add_parser('train', help='train the model')
    subparser.add_argument('--input', '-i', required=True, help='dataset JSON path')
    subparser.add_argument('--output', '-o', required=True, help='the model path to output')

    subparser = subparsers.add_parser('predict', help='predict using the trained mode')
    subparser.add_argument('--model', '-m', required=True, help='the model path')
    subparser.add_argument('--input', '-i', required=True, help='the path of .markermap file')
    subparser.add_argument('--dryrun', '-d', action='store_true', help='do not modify the .markermap file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.command == 'dataset':
        CreateDataset(markermapFolder=Path(args.input), outputPath=Path(args.output))
    elif args.command == 'train':
        pass
    elif args.command == 'predict':
        pass
