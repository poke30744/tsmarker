import json, logging, shutil, subprocess, copy
from pathlib import Path
from typing import Tuple
import numpy as np
from tscutter.common import ClipToFilename, PtsMap

logger = logging.getLogger('tsmarker.common')

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
        winThumbsPreloaderPath = Path(r'C:\Program Files\WinThumbsPreloader\WinThumbsPreloader.exe')
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


def _auto_by_method(marker_map: MarkerMap) -> str:
    props = marker_map.Properties()
    if '_groundtruth' in props:
        return '_groundtruth'
    elif '_ensemble' in props:
        return '_ensemble'
    else:
        return 'subtitles'


def _merge_neighbors(clips: list) -> list:
    merged = []
    for clip in clips:
        if not merged:
            merged.append(list(clip))
        elif merged[-1][1] == clip[0]:
            merged[-1][1] = clip[1]
        else:
            merged.append(list(clip))
    return [tuple(c) for c in merged]


def _clips_duration(clips: list) -> float:
    return sum(clip[1] - clip[0] for clip in clips)


def _split_clips(clips: list, num: int) -> list[list]:
    program_clips = list(clips)
    mean_duration = _clips_duration(program_clips) / num
    groups = []
    for i in range(num):
        group = []
        min_distance = mean_duration
        while program_clips:
            group.append(program_clips.pop(0))
            distance = abs(_clips_duration(group) - mean_duration)
            if distance >= min_distance:
                program_clips.insert(0, group.pop())
                break
            else:
                min_distance = distance
        groups.append(group)
    groups[-1].extend(program_clips)
    return groups


def get_program_clips(marker_path: Path, ptsmap_path: Path, by: str = 'auto', split: int = 1, by_group: bool = False) -> dict:
    ptsmap = PtsMap(ptsmap_path)
    marker_map = MarkerMap(marker_path, ptsmap)

    method = by if by != 'auto' else _auto_by_method(marker_map)
    all_clips = marker_map.Clips()
    program_clips = [clip for clip in all_clips if marker_map.Value(clip, method) > 0.5]

    merged = _merge_neighbors(program_clips)

    if by_group:
        groups = [[clip] for clip in merged]
    elif split > 1:
        groups = _split_clips(merged, split)
    else:
        groups = [merged]

    return {'groups': groups, 'by_method': method}

