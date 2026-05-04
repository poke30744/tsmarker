import sys
from pathlib import Path
from tqdm import tqdm


def ClipToFilename(clip: tuple[float, float]) -> str:
    return '{:08.3f}-{:08.3f}.ts'.format(float(clip[0]), float(clip[1]))


def _copy_bytes(src: Path, dst, start: int, length: int, pbar=None, bufsize: int = 1024 * 1024):
    with open(src, 'rb') as f:
        f.seek(start)
        while length:
            chunk = min(bufsize, length)
            data = f.read(chunk)
            dst.write(data)
            length -= chunk
            if pbar is not None:
                pbar.update(chunk)


def extract_clips_stdout(ts_path: Path, ptsmap_data: dict, clips: list[tuple[float, float]], quiet: bool = False):
    total_size = 0
    ranges: list[tuple[int, int]] = []
    for clip in clips:
        start_pos = ptsmap_data[str(clip[0])]['next_start_pos']
        end_pos = ptsmap_data[str(clip[1])]['prev_end_pos']
        ranges.append((start_pos, end_pos))
        total_size += end_pos - start_pos

    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, disable=quiet) as pbar:
        for start_pos, end_pos in ranges:
            _copy_bytes(ts_path, sys.stdout.buffer, start_pos, end_pos - start_pos, pbar=pbar)
    sys.stdout.buffer.flush()


def extract_clips_to_dir(ts_path: Path, ptsmap_data: dict, clips: list[tuple[float, float]], output_dir: Path, quiet: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    for clip in tqdm(clips, desc='Extracting clips', disable=quiet):
        start_pos = ptsmap_data[str(clip[0])]['next_start_pos']
        end_pos = ptsmap_data[str(clip[1])]['prev_end_pos']
        out_path = output_dir / ClipToFilename(clip)
        with open(out_path, 'wb') as dst:
            _copy_bytes(ts_path, dst, start_pos, end_pos - start_pos)
