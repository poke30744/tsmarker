import subprocess, argparse, logging
from pathlib import Path
import pysubs2
from tscutter.common import TsFileNotFound
from . import common

logger = logging.getLogger('tsmarker.subtitles')

def Extract(path) -> list:
    path = Path(path).expanduser()
    if not path.is_file():
        raise TsFileNotFound(f'"{path.name}" not found!')
    subExts = ( '.srt','.ass' ) if path.suffix == '.ts' else ( f'{path.suffix}.srt',f'{path.suffix}.ass' )
    subtitlesPathes = [ path.with_suffix(ext) for ext in subExts ]
    for p in subtitlesPathes:
        if p.exists():
            p.unlink()
    retry = 0
    while any([ not path.exists() for path in subtitlesPathes ]) and retry < 2:
        startupinfo = subprocess.STARTUPINFO(wShowWindow=6, dwFlags=subprocess.STARTF_USESHOWWINDOW) if hasattr(subprocess, 'STARTUPINFO') else None
        creationflags = subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
        pipeObj = subprocess.Popen(
            f'Captain2AssC.cmd "{path.absolute()}"',
            startupinfo=startupinfo,
            creationflags=creationflags,
            shell=True)
        pipeObj.wait()
        retry += 1
    availableSubs = []
    for p in subtitlesPathes:
        if p.exists():
            if p.stat().st_size == 0:
                p.unlink()
            else:
                # trying to fix syntax issues of Caption2AssC.exe
                subtitles = pysubs2.load(p, encoding='utf-8')
                subtitles.save(p)
            if path.suffix != '.ts':
                availableSubs.append(p.replace(p.with_name(p.name.replace(path.suffix, ''))))
            else:
                availableSubs.append(p)
    return availableSubs

def Overlap(range1, range2):
    return (range1[0] <= range2[0] <= range1[1]) or (range2[0] <= range1[0] <= range2[1])

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path) -> None:
        subs = Extract(videoPath)
        for path in subs:
            if path.with_suffix('.ass'):
                subtitles = pysubs2.load(path, encoding='utf-8')
                for clip in self.Clips():
                    overlap = False
                    for event in subtitles:
                        if Overlap((event.start / 1000, event.end / 1000), (clip[0], clip[1])):
                            overlap = True
                            break
                    self.Mark(clip, 'subtitles', 1.0 if overlap else 0.0)
            path.unlink()
        if len(subs) == 0:
            for clip in self.Clips():
                self.Mark(clip, 'subtitles', 0.5)
        self.Save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dump subtitles from TS file')
    parser.add_argument('--quiet', '-q', action='store_true', help="don't output to the console")
    parser.add_argument('--input', '-i', required=True, help='input mpegts path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    files = Extract(args.input)
    for path in files:
        print(path.name)