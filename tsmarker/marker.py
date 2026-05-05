import json, logging, sys
from pathlib import Path
import click
from rich.logging import RichHandler
from tscutter.common import PtsMap
from ._progress import Progress
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

def MarkVideo(videoPath, indexPath, markerPath, methods, progress=None, logoPath=None):
    videoPath = Path(videoPath)
    indexPath = Path(indexPath) if indexPath else videoPath.parent / '_metadata' / (videoPath.stem + '.ptsmap')
    markerPath = Path(markerPath) if markerPath else videoPath.parent / '_metadata' / (videoPath.stem + '.markermap')
    indexPath.parent.mkdir(parents=True, exist_ok=True)
    markerPath.parent.mkdir(parents=True, exist_ok=True)
    ptsMap = PtsMap(indexPath)
    for method in methods:
        if method == 'subtitles':
            subtitles.MarkerMap(markerPath, ptsMap).MarkAll(videoPath, progress=progress)
        elif method == 'logo':
            logo.MarkerMap(markerPath, ptsMap).MarkAll(videoPath, logoPath=Path(logoPath) if logoPath else None, progress=progress)
        elif method == 'clipinfo':
            clipinfo.MarkerMap(markerPath, ptsMap).MarkAll(videoPath, progress=progress)
        elif method == 'speech':
            speech.MarkerMap(markerPath, ptsMap).MarkAll(videoPath, progress=progress)
    return markerPath

@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.option('--progress', is_flag=True, help='Output PROGRESS JSON lines for pipeline orchestration')
@click.pass_context
def cli(ctx, quiet, progress):
    """Mark CMs in MPEG-TS files and manage the marker pipeline."""
    log_level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=log_level, format='%(message)s', datefmt='[%X]',
        handlers=[RichHandler(rich_tracebacks=True)])
    ctx.ensure_object(dict)
    ctx.obj['progress'] = Progress(use_protocol=progress)


@cli.command()
@click.option('--method', required=True, multiple=True,
              type=click.Choice(['subtitles', 'logo', 'clipinfo', 'speech']),
              help='Method(s) to mark CM')
@click.option('--input', '-i', required=True, help='Input mpegts path')
@click.option('--index', help='Mpegts index path (.ptsmap)')
@click.option('--marker', help='Output marker file path (.markermap)')
@click.option('--logo', help='Logo image path')
@click.pass_context
def mark(ctx, method, input, index, marker, logo):
    """Mark CM clips in the mpegts file using specified detection methods."""
    MarkVideo(videoPath=input, indexPath=index, markerPath=marker,
              methods=list(method), progress=ctx.obj['progress'], logoPath=logo)


@cli.command()
@click.option('--by', '-b', default='auto', show_default=True, help='Method to cut CMs by')
@click.option('--input', '-i', required=True, help='Input mpegts path')
@click.option('--index', help='Mpegts index path (.ptsmap)')
@click.option('--marker', help='Marker file path (.markermap)')
@click.option('--output', '-o', help='Output folder path')
@click.pass_context
def cut(ctx, by, input, index, marker, output):
    """Cut CMs from the mpegts file."""
    videoPath = Path(input)
    ptsPath = Path(index) if index is not None else videoPath.parent / '_metadata' / videoPath.with_suffix('.ptsmap').name
    markerPath = Path(marker) if marker is not None else videoPath.parent / '_metadata' / videoPath.with_suffix('.markermap').name
    outputFolder = Path(output) if output is not None else videoPath.with_suffix('')
    ptsMap = PtsMap(ptsPath)
    markerMap = MarkerMap(markerPath, ptsMap)
    by_method = by if by != 'auto' else _auto_by_method(markerMap)
    markerMap.Cut(videoPath=videoPath, byMethod=by_method, outputFolder=outputFolder, progress=ctx.obj['progress'])


@cli.command(name='groundtruth')
@click.option('--input', '-i', required=True, help='Input mpegts path')
@click.option('--index', help='Mpegts index path (.ptsmap)')
@click.option('--marker', help='Output marker file path (.markermap)')
@click.option('--clips', '-c', help='Clips folder')
@click.pass_context
def groundtruth_cmd(ctx, input, index, marker, clips):
    """Update groundtruth in .markermap after manual adjustment."""
    videoPath = Path(input)
    ptsPath = Path(index) if index else videoPath.parent / '_metadata' / (videoPath.stem + '.ptsmap')
    markerPath = Path(marker) if marker else videoPath.parent / '_metadata' / (videoPath.stem + '.markermap')
    clipsFolder = Path(clips) if clips else videoPath.with_suffix('')
    reEncodeNeeded = groundtruth.MarkerMap(markerPath, PtsMap(ptsPath)).MarkAll(clipsFolder)
    print(json.dumps({'re_encode_needed': reEncodeNeeded}))


@cli.command()
@click.option('--input', '-i', required=True, help='Input mpegts path')
@click.option('--index', '-x', required=True, help='.ptsmap file path')
@click.option('--clips', '-c', required=True, help='Clips JSON array, e.g. \'[[0.0,100.5],[200.0,350.8]]\'')
@click.option('--output-dir', '-o', help='Output directory for individual .ts files (stdout if omitted)')
@click.pass_context
def extract_clips(ctx, input, index, clips, output_dir):
    """Extract TS clips by byte range from .ptsmap."""
    ts_path = Path(input)
    pts_path = Path(index)
    try:
        pts_data = pts_path.read_bytes()
        ptsmap = json.loads(pts_data)
    except FileNotFoundError:
        print(f'FileNotFoundError: {index}', file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, KeyError):
        print(f'InvalidIndexFormat: {index}', file=sys.stderr)
        sys.exit(2)
    clips_data = json.loads(clips)
    if output_dir:
        extract_clips_to_dir(ts_path, ptsmap, clips_data, Path(output_dir), progress=ctx.obj['progress'])
    else:
        extract_clips_stdout(ts_path, ptsmap, clips_data, progress=ctx.obj['progress'])


@cli.command(name='get-program-clips')
@click.option('--marker', '-m', required=True, help='.markermap file path')
@click.option('--index', '-x', required=True, help='.ptsmap file path')
@click.option('--by', '-b', default='auto', show_default=True, help='Selection method: auto, _groundtruth, _ensemble, subtitles')
@click.option('--split', '-s', type=int, default=1, show_default=True, help='Split into N groups')
@click.option('--by-group', is_flag=True, help='Each clip as its own group')
def get_program_clips_cmd(marker, index, by, split, by_group):
    """Get program clips grouping from .markermap + .ptsmap."""
    result = get_program_clips(
        marker_path=Path(marker),
        ptsmap_path=Path(index),
        by=by,
        split=split,
        by_group=by_group,
    )
    print(json.dumps(result))


@cli.command()
@click.option('--input', '-i', required=True, help='Input mpegts path')
@click.option('--index', '-x', required=True, help='.ptsmap file path')
@click.option('--output', '-o', required=True, help='Output logo PNG path')
@click.option('--max-time', type=float, default=120, show_default=True, help='Max extraction time in seconds')
@click.option('--no-remove-border', is_flag=True, help='Do not remove frame border')
@click.pass_context
def extract_logo(ctx, input, index, output, max_time, no_remove_border):
    """Extract logo edge image from TS + .ptsmap."""
    ExtractLogoPipeline(
        inFile=Path(input),
        ptsMap=PtsMap(Path(index)),
        outFile=Path(output),
        maxTimeToExtract=max_time,
        removeBoarder=not no_remove_border,
        progress=ctx.obj['progress'],
    )


@cli.command()
@click.option('--input', '-i', required=True, help='Input logo PNG path')
@click.option('--threshold', '-t', type=float, default=0.3, show_default=True, help='Edge threshold')
def crop_detect(input, threshold):
    """Detect crop parameters from logo PNG."""
    crop = CropDetectPipeline(input, threshold=threshold)
    if crop is not None:
        crop = {k: int(v) for k, v in crop.items()}
    print(json.dumps(crop))


@cli.command()
@click.option('--input', '-i', required=True, help='Input mpegts path')
@click.option('--index', '-x', required=True, help='.ptsmap file path')
@click.pass_context
def prepare_subtitles(ctx, input, index):
    """Extract subtitles and generate speech-to-text."""
    PrepareSubtitles(Path(input), PtsMap(Path(index)), progress=ctx.obj['progress'])


@cli.command()
@click.option('--input', '-i', required=True, help='Search folder for .mp4 + .yaml + .markermap')
@click.option('--output', '-o', required=True, help='Output CSV path')
@click.option('--no-normalize', is_flag=True, help='Skip normalization')
@click.pass_context
def ensemble_dataset(ctx, input, output, no_normalize):
    """Generate ensemble training dataset CSV."""
    import pandas as pd
    df = ensemble.CreateDataset(
        folder=Path(input),
        csvPath=Path(output),
        normalize=not no_normalize,
        progress=ctx.obj['progress'],
    )
    if df is not None:
        logger.info(f'Dataset created: {output}')
    else:
        logger.warning(f'No metadata found in {input}!')


@cli.command()
@click.option('--input', '-i', required=True, help='Input CSV path')
@click.option('--output', '-o', required=True, help='Output model path')
@click.option('--random-state', type=int, default=0, show_default=True, help='Random state')
@click.option('--test-size', type=float, default=0.3, show_default=True, help='Test set size ratio')
@click.pass_context
def ensemble_train(ctx, input, output, random_state, test_size):
    """Train ensemble model from dataset CSV."""
    dataset = ensemble.LoadDataset(csvPath=Path(input))
    columns = dataset['columns']
    clf = ensemble.Train(dataset, random_state=random_state, test_size=test_size, progress=ctx.obj['progress'])
    import pickle
    with open(output, 'wb') as f:
        pickle.dump((clf, columns), f)
    logger.info(f'Model saved to {output}')


@cli.command()
@click.option('--model', '-m', required=True, help='Model file path')
@click.option('--index', '-x', required=True, help='.ptsmap file path')
@click.option('--marker', required=True, help='.markermap file path')
@click.option('--normalize', is_flag=True, help='Normalize before prediction')
@click.option('--dry-run', is_flag=True, help='Only print, do not write')
def ensemble_predict(model, index, marker, normalize, dry_run):
    """Predict using trained ensemble model."""
    import pickle
    with open(model, 'rb') as f:
        clf, columns = pickle.load(f)
    model_tuple = (clf, columns)
    marker_path = Path(marker)
    ptsmap = PtsMap(Path(index))
    ensemble.MarkerMap(marker_path, ptsmap).MarkAll(model_tuple, normalize=normalize)
    if dry_run:
        logger.info('Dry run — no changes written')


def main():
    cli()


if __name__ == "__main__":
    main()