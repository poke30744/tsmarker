import json, shutil, subprocess, copy
from pathlib import Path
from typing import Tuple
import numpy as np
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
    
    def Cut(self, videoPath: Path, byMethod: str, outputFolder: Path, quiet: bool=False) -> None:
        cmFolder = outputFolder / 'CM'
        cmMoveList = []
        programList = []
        for clip in self.Clips():
            if self.data[str(clip)][byMethod] < 0.5:
                cmMoveList.append(clip)
            else:
                programList.append(clip)
        self.ptsMap.SplitVideo(videoPath=videoPath, outputFolder=outputFolder, quiet=quiet)
        cmFolder = outputFolder / 'CM'
        cmFolder.mkdir()
        for clip in cmMoveList:
            shutil.move(outputFolder / ClipToFilename(clip), cmFolder / self.ClipToFilenameForReview(clip))
        for clip in programList:
            shutil.move(outputFolder / ClipToFilename(clip), outputFolder / self.ClipToFilenameForReview(clip))
        
        # pre-load thrumbs
        winThumbsPreloaderPath = Path('C:\Program Files\WinThumbsPreloader\WinThumbsPreloader.exe')
        if winThumbsPreloaderPath.exists():
            subprocess.call(f'{winThumbsPreloaderPath} -r "{outputFolder}"')
    
    def Normalized(self) -> dict:
        properties = self.Properties()
        normalized = copy.deepcopy(self.data)
        for prop in properties:
            if not prop in ('_ensemble', '_groundtruth', 'position', 'duration', 'duration_prev', 'duration_next'):
                raw = [ self.data[k][prop] for k in self.data.keys() ]
                mean = np.mean(raw)
                std = np.std(raw)
                for k in normalized.keys():
                    normalized[k][prop] -= mean
                    if std != 0:
                        normalized[k][prop] /= std
        return normalized

    def ClipToFilenameForReview(self, clip: Tuple[float, float]) -> str:
        hasLogo = self.data[str(clip)]["logo"]
        hasSubtitles = self.data[str(clip)]["subtitles"]
        name = Path(ClipToFilename(clip))
        newStem = name.stem
        if hasLogo > 0.5:
            newStem += '.L'
        if hasSubtitles > 0.5:
            newStem += '.S'
        return str(name.with_stem(newStem))

