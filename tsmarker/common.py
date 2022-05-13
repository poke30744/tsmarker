import json, shutil, subprocess
from pathlib import Path
from tscutter.common import ClipToFilename, PtsMap

class MarkerMap:    
    def __init__(self, path: Path, ptsMap: PtsMap) -> None:
        self.path = path
        self.ptsMap = ptsMap
        if not self.path.exists():
            self.data = { str(clip) : {} for clip in ptsMap.Clips() }
        else:
            with self.path.open() as f:
                self.data = json.load(f)
    
    def Properties(self) -> list:
        return list(list(self.data.items())[0][1].keys())
    
    def Mark(self, clip, property, value) -> None:
        self.data[str(clip)][property] = value
    
    def Value(self, clip, property) -> float:
        return self.data[str(clip)][property]
    
    def Save(self) -> None:
         with self.path.open('w') as f:
            json.dump(self.data, f, indent=True)
    
    def Clips(self) -> list:
        return self.ptsMap.Clips()
    
    def Cut(self, videoPath: Path, byMethod: str, outputFolder: Path) -> None:
        cmFolder = outputFolder / 'CM'
        cmMoveList = []
        programList = []
        for clip in self.Clips():
            clipFilename = ClipToFilename(clip)
            if self.data[str(clip)][byMethod] < 0.5:
                cmMoveList.append((outputFolder / clipFilename, cmFolder / clipFilename))
            else:
                programList.append(outputFolder / clipFilename)
        self.ptsMap.SplitVideo(videoPath=videoPath, outputFolder=outputFolder)
        cmFolder = outputFolder / 'CM'
        cmFolder.mkdir()
        for src, dst in cmMoveList:
            shutil.move(src, dst)
        
        # pre-load thrumbs
        winThumbsPreloaderPath = Path('C:\Program Files\WinThumbsPreloader\WinThumbsPreloader.exe')
        if winThumbsPreloaderPath.exists():
            subprocess.call(f'{winThumbsPreloaderPath} -r "{outputFolder}"')

