import subprocess, argparse, logging
from pathlib import Path
import tempfile
import pysubs2
from tscutter.common import CopyPartPipe, TsFileNotFound
from . import common

logger = logging.getLogger('tsmarker.subtitles')

def Extract(path: Path, folder: Path) -> list[Path]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise TsFileNotFound(f'"{path.name}" not found!')

    subtitlesPathes = [ (folder / path.stem).with_suffix(ext) for ext in ( '.srt','.ass' ) ]

    # remove existing subtitles files
    for p in subtitlesPathes:
        if p.exists():
            p.unlink()
    
    # use a short ASCII name to avoid encoding errors
    subtitlesStem = 'tsmarker_subtitles'

    retry = 0
    availableSubs = []
    while len(availableSubs) == 0 and retry < 2:
        startupinfo = subprocess.STARTUPINFO(wShowWindow=6, dwFlags=subprocess.STARTF_USESHOWWINDOW) if hasattr(subprocess, 'STARTUPINFO') else None
        creationflags = subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
        pipeObj = subprocess.Popen(
            f'Caption2AssC.cmd - "{folder.absolute() / subtitlesStem}"',
            stdin=subprocess.PIPE,
            startupinfo=startupinfo,
            creationflags=creationflags,
            shell=True)
        try:
            CopyPartPipe(path, pipeObj.stdin, 0, path.stat().st_size)
        except BrokenPipeError:
            pass
        pipeObj.stdin.close()
        pipeObj.wait()
        
        # rename back
        for p in folder.glob('*'):
            if p.stem == subtitlesStem:
                p.rename(p.with_stem(path.stem))

        # fix subtitles
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
        retry += 1
    return availableSubs

def Overlap(range1, range2):
    return (range1[0] <= range2[0] <= range1[1]) or (range2[0] <= range1[0] <= range2[1])

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, assPath: Path=None) -> None:
        subtitles = None
        with tempfile.TemporaryDirectory(prefix='ExtractSubtitles_') as tmpFolder:
            for p in Extract(videoPath, Path(tmpFolder)):
                if p.with_suffix('.ass'):
                    subtitles = pysubs2.load(p, encoding='utf-8')
                    break
        if subtitles is not None:
            for clip in self.Clips():
                overlap = False
                for event in subtitles:
                    if Overlap((event.start / 1000, event.end / 1000), (clip[0], clip[1])):
                        overlap = True
                        break
                self.Mark(clip, 'subtitles', 1.0 if overlap else 0.0)
        else:
            for clip in self.Clips():
                self.Mark(clip, 'subtitles', 0.5)
        self.Save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dump subtitles from TS file')
    parser.add_argument('--quiet', '-q', action='store_true', help="don't output to the console")
    parser.add_argument('--input', '-i', required=True, help='input mpegts path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    path = Path(args.input)
    files = Extract(path, path.parent)
    for path in files:
        print(path.name)