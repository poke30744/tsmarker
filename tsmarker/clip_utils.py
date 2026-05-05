import sys
from pathlib import Path


def ClipToFilename(clip: tuple[float, float]) -> str:
    return '{:08.3f}-{:08.3f}.ts'.format(float(clip[0]), float(clip[1]))


def extract_clips_stdout(ts_path: Path, ptsmap_data: dict, clips: list[tuple[float, float]], progress=None):
    total_size = 0
    ranges: list[tuple[int, int]] = []
    for clip in clips:
        start_pos = ptsmap_data[str(clip[0])]['next_start_pos']
        end_pos = ptsmap_data[str(clip[1])]['prev_end_pos']
        ranges.append((start_pos, end_pos))
        total_size += end_pos - start_pos

    tid = "extract_clips"
    if progress is not None:
        progress.add_task(tid, total_size, "Extracting clips", unit="B")
    copied = 0
    bufsize = 1024 * 1024
    for start_pos, end_pos in ranges:
        length = end_pos - start_pos
        with open(ts_path, 'rb') as f:
            f.seek(start_pos)
            while length:
                chunk = min(bufsize, length)
                data = f.read(chunk)
                sys.stdout.buffer.write(data)
                length -= chunk
                copied += chunk
                if progress is not None:
                    progress.update(tid, copied)
    sys.stdout.buffer.flush()
    if progress is not None:
        progress.done(tid)


def extract_clips_to_dir(ts_path: Path, ptsmap_data: dict, clips: list[tuple[float, float]], output_dir: Path, progress=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    tid = "extract_clips"
    if progress is not None:
        progress.add_task(tid, len(clips), "Extracting clips")
    for i, clip in enumerate(clips):
        start_pos = ptsmap_data[str(clip[0])]['next_start_pos']
        end_pos = ptsmap_data[str(clip[1])]['prev_end_pos']
        out_path = output_dir / ClipToFilename(clip)
        with open(out_path, 'wb') as dst:
            with open(ts_path, 'rb') as f:
                f.seek(start_pos)
                length = end_pos - start_pos
                bufsize = 1024 * 1024
                while length:
                    chunk = min(bufsize, length)
                    dst.write(f.read(chunk))
                    length -= chunk
        if progress is not None:
            progress.update(tid, i + 1)
    if progress is not None:
        progress.done(tid)
