import argparse, shutil, json, sys
from pathlib import Path
from .common import LoadExistingData, SaveMarkerMap, GroundTruthError
from tsutils.common import ClipToFilename
from tscutter.analyze import SplitVideo
import tsmarker.subtitles
import tsmarker.clipinfo
import tsmarker.logo

def MarkVideo(videoPath, indexPath, markerPath, methods, quiet=False):
    videoPath = Path(videoPath)
    indexPath = Path(indexPath) if indexPath else videoPath.parent / '_metadata' / (videoPath.stem + '.ptsmap')
    markerPath = Path(markerPath) if markerPath else videoPath.parent / '_metadata' / (videoPath.stem + '.markermap')
    indexPath.parent.mkdir(parents=True, exist_ok=True)
    markerPath.parent.mkdir(parents=True, exist_ok=True)
    if markerPath.exists() and markerPath.stat().st_mtime > indexPath.stat().st_mtime:
        print(f'Skipping marking for {videoPath.name}', file=sys.stderr)
        return markerPath
    for method in methods:
        if method == 'subtitles': 
            markerPath =  tsmarker.subtitles.Mark(videoPath=videoPath, indexPath=indexPath, markerPath=markerPath)
        elif method == 'logo':
            markerPath =  tsmarker.logo.Mark(videoPath=videoPath, indexPath=indexPath, markerPath=markerPath, quiet=quiet)
        elif method == 'clipinfo':
            markerPath =  tsmarker.clipinfo.Mark(videoPath=videoPath, indexPath=indexPath, markerPath=markerPath, quiet=quiet)
    if markerPath.exists():
        markerPath.touch()
    return markerPath

def CutCMs(videoPath, indexPath, markerPath, byMethod, outputFolder, quiet=False):
    videoPath = Path(videoPath)
    indexPath = Path(indexPath) if indexPath else  videoPath.parent / '_metadata' / (videoPath.stem + '.ptsmap')
    markerPath = Path(markerPath) if markerPath else videoPath.parent / '_metadata' / (videoPath.stem + '.markermap')
    outputFolder = Path(outputFolder) if outputFolder else videoPath.with_suffix('')
    _, markerMap = LoadExistingData(indexPath, markerPath)
    cmFolder = outputFolder / 'CM'
    cmMoveList = []
    programList = []
    for clipStr in markerMap:
        clipFilename = ClipToFilename(eval(clipStr))
        if markerMap[clipStr][byMethod] < 0.5:
            cmMoveList.append((outputFolder / clipFilename, cmFolder / clipFilename))
        else:
            programList.append(outputFolder / clipFilename)
    if all([ srcdst[1].exists() for srcdst in cmMoveList ]) and all([ path.exists() for path in programList ]):
        print(f'Skipping cutting CMs for {videoPath.name}', file=sys.stderr)
        for path in outputFolder.glob('**/*.ts'):
            shutil.copystat(markerPath, path)
        return outputFolder

    SplitVideo(videoPath, indexPath, outputFolder, quiet)
    cmFolder = outputFolder / 'CM'
    cmFolder.mkdir()
    for src, dst in cmMoveList:
        shutil.copystat(markerPath, src)
        shutil.move(src, dst)
    return outputFolder

def MarkGroundTruth(clipsFolder, markerPath):
    clipsFolder = Path(clipsFolder)
    cmFolder = clipsFolder / 'CM'
    markerPath = Path(markerPath) if markerPath else clipsFolder.parent / '_metadata' / (clipsFolder.stem + '.ptsmap')
    with markerPath.open() as f:
        markerMap = json.load(f)
    changed = False
    for clipStr in markerMap:
        clipFilename = ClipToFilename(eval(clipStr))
        if '_groundtruth' in markerMap[clipStr]:
            existingGT = markerMap[clipStr]['_groundtruth']
        elif '_ensemble' in markerMap[clipStr]:
            existingGT = markerMap[clipStr]['_ensemble']
        else:
            existingGT = None
        if (clipsFolder / clipFilename).exists():
            groundTruth = 1.0
        elif (cmFolder / clipFilename).exists():
            groundTruth = 0.0
        else:
            raise GroundTruthError(f'{clipStr} not exist in {clipsFolder}!')
        markerMap[clipStr]['_groundtruth'] = groundTruth
        if groundTruth != existingGT:
            changed = True
    if not changed:
        oldMarkerPath = markerPath.rename(markerPath.with_suffix('.m1'))
    markerPath = SaveMarkerMap(markerMap, markerPath)
    if not changed:
        shutil.copystat(oldMarkerPath, markerPath)
        oldMarkerPath.unlink()
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
    subparser.add_argument('--input', '-i', required=True, help='the folder contains CM/')
    subparser.add_argument('--output', '-o', help='output marker file path (.markermap)')

    args = parser.parse_args()

    if args.command == 'mark':
        MarkVideo(videoPath=args.input, indexPath=args.index, markerPath=args.marker, methods=args.method, quiet=args.quiet)
    elif args.command == 'cut':
        CutCMs(videoPath=args.input, indexPath=args.index, markerPath=args.marker, byMethod=args.method, outputFolder=args.output)
    elif args.command == 'groundtruth':
        MarkGroundTruth(clipsFolder=args.input, markerPath=args.output)