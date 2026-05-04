import argparse, json, logging, sys
from pathlib import Path
from tscutter.common import PtsMap
from . import subtitles
from . import clipinfo
from . import logo
from . import speech
from . import groundtruth
from .common import MarkerMap, get_program_clips, _auto_by_method
from .clip_utils import extract_clips_stdout, extract_clips_to_dir
from .pipeline import ExtractLogoPipeline, CropDetectPipeline
from .speech.text_extractor import PrepareSubtitles
from . import ensemble

logger = logging.getLogger('tsmarker.marker')

def MarkVideo(videoPath, indexPath, markerPath, methods, quiet=False, logoPath=None):
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
            logo.MarkerMap(markerPath, ptsMap).MarkAll(videoPath, logoPath=Path(logoPath) if logoPath else None, quiet=quiet)
        elif method == 'clipinfo':
            clipinfo.MarkerMap(markerPath, ptsMap).MarkAll(videoPath, quiet=quiet)
        elif method == 'speech':
            speech.MarkerMap(markerPath, ptsMap).MarkAll(videoPath)
    return markerPath

def main():
    parser = argparse.ArgumentParser(description='Python tool to mark CMs in mpegts')
    
    parser.add_argument('--quiet', '-q', action='store_true', help="don't output to the console")

    subparsers = parser.add_subparsers(required=True, title='subcommands', dest='command')

    subparser = subparsers.add_parser('mark', help='mark CM clips in the mpegts file')
    subparser.add_argument('--method', required=True, nargs='+', choices=['subtitles', 'logo', 'clipinfo', 'speech'], help='method to mark CM')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--index', help='mpegts index path (.ptsmap)')
    subparser.add_argument('--marker', help='output marker file path (.markermap)')
    subparser.add_argument('--logo', help='logo image path')

    subparser = subparsers.add_parser('cut', help='cut CMs from the mpegts file')
    subparser.add_argument('--by', '-b', default='auto', help='by which method to cut CMs (default: auto)')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--index', help='mpegts index path (.ptsmap)')
    subparser.add_argument('--marker', help='marker file path (.markermap)')
    subparser.add_argument('--output', '-o', help='output folder path')

    subparser = subparsers.add_parser('groundtruth', help='update groundtruth in .markermap after manual adjustment')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--index', help='mpegts index path (.ptsmap)')
    subparser.add_argument('--marker', help='output marker file path (.markermap)')
    subparser.add_argument('--clips', '-c', help='clips folder')

    subparser = subparsers.add_parser('extract-clips', help='extract TS clips by byte range from .ptsmap')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--index', '-x', required=True, help='.ptsmap file path')
    subparser.add_argument('--clips', '-c', required=True, help='clips JSON array, e.g. \'[[0.0,100.5],[200.0,350.8]]\'')
    subparser.add_argument('--output-dir', '-o', help='output directory for individual .ts files (stdout if omitted)')

    subparser = subparsers.add_parser('get-program-clips', help='get program clips grouping from .markermap + .ptsmap')
    subparser.add_argument('--marker', '-m', required=True, help='.markermap file path')
    subparser.add_argument('--index', '-x', required=True, help='.ptsmap file path')
    subparser.add_argument('--by', '-b', default='auto', help='selection method: auto, _groundtruth, _ensemble, subtitles')
    subparser.add_argument('--split', '-s', type=int, default=1, help='split into N groups')
    subparser.add_argument('--by-group', action='store_true', default=False, help='each clip as its own group')

    subparser = subparsers.add_parser('extract-logo', help='extract logo edge image from TS + .ptsmap')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--index', '-x', required=True, help='.ptsmap file path')
    subparser.add_argument('--output', '-o', required=True, help='output logo PNG path')
    subparser.add_argument('--max-time', type=float, default=120, help='max extraction time in seconds')
    subparser.add_argument('--no-remove-border', action='store_true', help='do not remove frame border')

    subparser = subparsers.add_parser('crop-detect', help='detect crop parameters from logo PNG')
    subparser.add_argument('--input', '-i', required=True, help='input logo PNG path')
    subparser.add_argument('--threshold', '-t', type=float, default=0.3, help='edge threshold')

    subparser = subparsers.add_parser('prepare-subtitles', help='extract subtitles and generate speech-to-text')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--index', '-x', required=True, help='.ptsmap file path')

    subparser = subparsers.add_parser('ensemble-dataset', help='generate ensemble training dataset')
    subparser.add_argument('--input', '-i', required=True, help='search folder for .mp4 + .yaml + .markermap')
    subparser.add_argument('--output', '-o', required=True, help='output CSV path')
    subparser.add_argument('--no-normalize', action='store_true', help='skip normalization')

    subparser = subparsers.add_parser('ensemble-train', help='train ensemble model')
    subparser.add_argument('--input', '-i', required=True, help='input CSV path')
    subparser.add_argument('--output', '-o', required=True, help='output model path')
    subparser.add_argument('--random-state', type=int, default=0, help='random state')
    subparser.add_argument('--test-size', type=float, default=0.3, help='test set size ratio')

    subparser = subparsers.add_parser('ensemble-predict', help='predict using ensemble model')
    subparser.add_argument('--model', '-m', required=True, help='model file path')
    subparser.add_argument('--index', '-x', required=True, help='.ptsmap file path')
    subparser.add_argument('--marker', required=True, help='.markermap file path')
    subparser.add_argument('--normalize', action='store_true', help='normalize before prediction')
    subparser.add_argument('--dry-run', action='store_true', help='only print, do not write')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.command == 'extract-logo':
        ExtractLogoPipeline(
            inFile=Path(args.input),
            ptsMap=PtsMap(Path(args.index)),
            outFile=Path(args.output),
            maxTimeToExtract=args.max_time,
            removeBoarder=not args.no_remove_border,
            quiet=args.quiet,
        )

    elif args.command == 'crop-detect':
        crop = CropDetectPipeline(args.input, threshold=args.threshold)
        if crop is not None:
            crop = {k: int(v) for k, v in crop.items()}
        print(json.dumps(crop))

    elif args.command == 'prepare-subtitles':
        PrepareSubtitles(Path(args.input), PtsMap(Path(args.index)), quiet=args.quiet)

    elif args.command == 'ensemble-dataset':
        import pandas as pd
        df = ensemble.CreateDataset(
            folder=Path(args.input),
            csvPath=Path(args.output),
            normalize=not args.no_normalize,
            quiet=args.quiet,
        )
        if df is not None:
            logger.info(f'Dataset created: {args.output}')
        else:
            logger.warning(f'No metadata found in {args.input}!')

    elif args.command == 'ensemble-train':
        dataset = ensemble.LoadDataset(csvPath=Path(args.input))
        columns = dataset['columns']
        clf = ensemble.Train(dataset, random_state=args.random_state, test_size=args.test_size, quiet=args.quiet)
        import pickle
        with open(args.output, 'wb') as f:
            pickle.dump((clf, columns), f)
        logger.info(f'Model saved to {args.output}')

    elif args.command == 'ensemble-predict':
        import pickle
        with open(args.model, 'rb') as f:
            clf, columns = pickle.load(f)
        model = (clf, columns)
        marker_path = Path(args.marker)
        ptsmap = PtsMap(Path(args.index))
        ensemble.MarkerMap(marker_path, ptsmap).MarkAll(model, normalize=args.normalize)
        if args.dry_run:
            logger.info('Dry run — no changes written')

    elif args.command == 'get-program-clips':
        result = get_program_clips(
            marker_path=Path(args.marker),
            ptsmap_path=Path(args.index),
            by=args.by,
            split=args.split,
            by_group=args.by_group,
        )
        print(json.dumps(result))

    elif args.command == 'extract-clips':
        ts_path = Path(args.input)
        pts_path = Path(args.index)
        try:
            pts_data = pts_path.read_bytes()
            ptsmap = json.loads(pts_data)
        except FileNotFoundError:
            print(f'FileNotFoundError: {args.index}', file=sys.stderr)
            sys.exit(1)
        except (json.JSONDecodeError, KeyError):
            print(f'InvalidIndexFormat: {args.index}', file=sys.stderr)
            sys.exit(2)
        clips = json.loads(args.clips)
        if args.output_dir:
            extract_clips_to_dir(ts_path, ptsmap, clips, Path(args.output_dir), quiet=args.quiet)
        else:
            extract_clips_stdout(ts_path, ptsmap, clips, quiet=args.quiet)

    elif args.command == 'mark':
        MarkVideo(videoPath=args.input, indexPath=args.index, markerPath=args.marker, methods=args.method, quiet=args.quiet, logoPath=args.logo)
    elif args.command == 'cut':
        videoPath = Path(args.input)
        ptsPath = Path(args.index) if args.index is not None else videoPath.parent / '_metadata' / videoPath.with_suffix('.ptsmap').name
        markerPath = Path(args.marker) if args.marker is not None else videoPath.parent / '_metadata' / videoPath.with_suffix('.markermap').name
        outputFolder = Path(args.output) if args.output is not None else videoPath.with_suffix('')
        ptsMap = PtsMap(ptsPath)
        markerMap = MarkerMap(markerPath, ptsMap)
        by_method = args.by if args.by != 'auto' else _auto_by_method(markerMap)
        markerMap.Cut(videoPath=videoPath, byMethod=by_method, outputFolder=outputFolder)
    elif args.command == 'groundtruth':
        videoPath = Path(args.input)
        ptsPath = Path(args.index) if args.index else videoPath.parent / '_metadata' / (videoPath.stem + '.ptsmap')
        markerPath = Path(args.marker) if args.marker else videoPath.parent / '_metadata' / (videoPath.stem + '.markermap')
        clipsFolder = Path(args.clips) if args.clips else videoPath.with_suffix('')
        reEncodeNeeded = groundtruth.MarkerMap(markerPath, PtsMap(ptsPath)).MarkAll(clipsFolder)
        print(json.dumps({'re_encode_needed': reEncodeNeeded}))

if __name__ == "__main__":
    main()