import argparse, logging
from pathlib import Path
from tscutter.common import PtsMap
from . import subtitles
from . import clipinfo
from . import logo
from .common import MarkerMap
from . import groundtruth

logger = logging.getLogger('tsmarker.marker')

def MarkVideo(videoPath, indexPath, markerPath, methods, quiet=False):
    videoPath = Path(videoPath)
    indexPath = Path(indexPath) if indexPath else videoPath.parent / '_metadata' / (videoPath.stem + '.ptsmap')
    markerPath = Path(markerPath) if markerPath else videoPath.parent / '_metadata' / (videoPath.stem + '.markermap')
    indexPath.parent.mkdir(parents=True, exist_ok=True)
    markerPath.parent.mkdir(parents=True, exist_ok=True)
    ptsMap = PtsMap(indexPath)
    for method in methods:
        if method == 'subtitles':
            subtitles.MarkerMap(markerPath, ptsMap).MarkAll(videoPath)
        elif method == 'logo':
            logo.MarkerMap(markerPath, ptsMap).MarkAll(videoPath, quiet=quiet)
        elif method == 'clipinfo':
            clipinfo.MarkerMap(markerPath, ptsMap).MarkAll(videoPath, quiet=quiet)
    return markerPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python tool to mark CMs in mpegts')
    
    parser.add_argument('--quiet', '-q', action='store_true', help="don't output to the console")

    subparsers = parser.add_subparsers(required=True, title='subcommands', dest='command')

    subparser = subparsers.add_parser('mark', help='mark CM clips in the mpegts file')
    subparser.add_argument('--method', required=True, nargs='+', choices=['subtitles', 'logo', 'clipinfo'], help='method to mark CM')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--index', help='mpegts index path (.ptsmap)')
    subparser.add_argument('--marker', help='output marker file path (.markermap)')

    subparser = subparsers.add_parser('cut', help='cut CMs from the mpegts file')
    subparser.add_argument('--method', required=True, help='by which method to cut CMs')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--index', help='mpegts index path (.ptsmap)')
    subparser.add_argument('--marker', help='marker file path (.markermap)')
    subparser.add_argument('--output', '-o', help='output folder path')

    subparser = subparsers.add_parser('groundtruth', help='update groundtruth in .markermap after manual adjustment')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--index', help='mpegts index path (.ptsmap)')
    subparser.add_argument('--marker', help='output marker file path (.markermap)')
    subparser.add_argument('--clips', '-c', help='clips folder')

    subparser = subparsers.add_parser('merge', help='merge mpegts files together with .ptsmap and .markermap')
    subparser.add_argument('--input', '-i', required=True, nargs='+', help='mpegts files to merge')

    args = parser.parse_args()

    if args.command == 'mark':
        MarkVideo(videoPath=args.input, indexPath=args.index, markerPath=args.marker, methods=args.method, quiet=args.quiet)
    elif args.command == 'cut':
        videoPath = Path(args.input)
        ptsPath = Path(args.index) if args.index is not None else videoPath.parent / '_metadata' / videoPath.with_suffix('.ptsmap').name
        markerPath = Path(args.marker) if args.marker is not None else videoPath.parent / '_metadata' / videoPath.with_suffix('.markermap').name
        outputFolder = Path(args.output) if args.output is not None else videoPath.with_suffix('')
        MarkerMap(markerPath, PtsMap(ptsPath)).Cut(videoPath=videoPath, byMethod=args.method, outputFolder=outputFolder)
    elif args.command == 'groundtruth':
        videoPath = Path(args.input)
        ptsPath = Path(args.index) if args.index else videoPath.parent / '_metadata' / (videoPath.stem + '.ptsmap')
        markerPath = Path(args.marker) if args.marker else videoPath.parent / '_metadata' / (videoPath.stem + '.markermap')
        clipsFolder = Path(args.clips) if args.clips else videoPath.with_suffix('')
        reEncodeNeeded = groundtruth.MarkerMap(markerPath, PtsMap(ptsPath)).MarkAll(clipsFolder)
        print(f'reEncodeNeeded: {reEncodeNeeded}')