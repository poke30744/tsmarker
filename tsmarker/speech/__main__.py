import argparse
import logging
from pathlib import Path
from .dataset import *
from .MarkerMap import ReMarkAll

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict clips using LLM')
    subparsers = parser.add_subparsers(required=True, title='subcommands', dest='command')

    subparser = subparsers.add_parser('dataset', help='create CSV dataset from .markermap files')
    subparser.add_argument('--input', '-i', required=True, help='the folder where to search *.markermap files')
    subparser.add_argument('--output', '-o', required=True, help='output JSON path')

    subparser = subparsers.add_parser('mark', help='mark clips using LLM')
    subparser.add_argument('--input', '-i', required=True, help='parent folder of .markermap files')
    subparser.add_argument('--url', '-u', default='', help='API URL (ignored, kept for compatibility)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.command == 'dataset':
        CreateDataset(markermapFolder=Path(args.input), outputPath=Path(args.output))
    elif args.command == 'mark':
        ReMarkAll(Path(args.input), args.url)