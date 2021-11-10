import subprocess, argparse, logging
from pathlib import Path
import pysubs2
from tscutter.common import TsFileNotFound
from .common import LoadExistingData, GetClips, SaveMarkerMap

logger = logging.getLogger('tsmarker.subtitles')

def Extract(path):
    path = Path(path)
    if not path.is_file():
        raise TsFileNotFound(f'"{path.name}" not found!')
    subtitlesPathes = [ Path(path).with_suffix(ext) for ext in ( '.srt','.ass' ) ]
    for subtitlePath in subtitlesPathes:
        if subtitlePath.exists():
            subtitlePath.unlink()
    retry = 0
    while any([ not path.exists() for path in subtitlesPathes ]) and retry < 2:
        pipeObj = subprocess.Popen(
            f'Captain2AssC.cmd "{path.absolute()}"',
            startupinfo=subprocess.STARTUPINFO(wShowWindow=6, dwFlags=subprocess.STARTF_USESHOWWINDOW),
            creationflags=subprocess.CREATE_NEW_CONSOLE)
        pipeObj.wait()
        retry += 1
    for subtitlesPath in subtitlesPathes:
        if subtitlesPath.exists():
            if subtitlesPath.stat().st_size == 0:
                subtitlesPath.unlink()
            else:
                # trying to fix syntax issues of Caption2AssC.exe
                subtitles = pysubs2.load(subtitlesPath, encoding='utf-8')
                subtitles.save(subtitlesPath)
    return [ path for path in subtitlesPathes if path.exists() ]

def Overlap(range1, range2):
    return (range1[0] <= range2[0] <= range1[1]) or (range2[0] <= range1[0] <= range2[1])

def Mark(videoPath, indexPath, markerPath):
    ptsMap, markerMap = LoadExistingData(indexPath=indexPath, markerPath=markerPath)
    Extract(videoPath)
    assPath = videoPath.with_suffix('.ass')
    if assPath.exists():
        subtitles = pysubs2.load(assPath, encoding='utf-8')
        for clip in GetClips(ptsMap):
            overlap = False
            for event in subtitles:
                if Overlap((event.start / 1000, event.end / 1000), (clip[0], clip[1])):
                    overlap = True
                    break
            markerMap[str(clip)]['subtitles'] = 1.0 if overlap else 0.0
        for suffix in ('.ass', '.srt'):
            path = videoPath.with_suffix(suffix)
            path.unlink()
    else:
        for clip in GetClips(ptsMap):
            markerMap[str(clip)]['subtitles'] = 0.5
    markerPath = SaveMarkerMap(markerMap, markerPath)
    return markerPath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dump subtitles from TS file')
    parser.add_argument('--quiet', '-q', action='store_true', help="don't output to the console")
    parser.add_argument('--input', '-i', required=True, help='input mpegts path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    files = Extract(args.input)
    for path in files:
        print(path.name)