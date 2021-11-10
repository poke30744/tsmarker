import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from tscutter.common import FormatTimestamp
from .marker import CutCMs

class GroundTruthError(RuntimeError): ...

def GetClips(ptsMap):
    return [ ( float(list(ptsMap.keys())[i]),  float(list(ptsMap.keys())[i + 1]) ) for i in range(len(ptsMap) - 1) ]

def LoadExistingData(indexPath, markerPath):
    with indexPath.open() as f:
        ptsMap = json.load(f)
    if markerPath.exists():
        with markerPath.open() as f:
            markerMap = json.load(f)
    else:
        markerMap = { str(clip) : {} for clip in GetClips(ptsMap) }
    return ptsMap, markerMap

def SaveMarkerMap(markerMap, markerPath):
    with markerPath.open('w') as f:
        json.dump(markerMap, f, indent=True)
    return markerPath

def SelectClips(clips, lengthLimit=150, durationLimit=0.5):
    videoLen = clips[-1][1]
    selectedClips = []
    selectedLen = 0
    for clip in reversed(sorted(clips, key=lambda clip: clip[1] - clip[0])):
        clipLen = clip[1] - clip[0]
        if clipLen < lengthLimit:
            break
        if selectedLen > videoLen / 2:
            break
        selectedClips.append(clip)
        selectedLen += clipLen
    return selectedClips, selectedLen

def RemoveBoarder(edges, threshold=0.2):
    shape = edges.shape
    threshold *= 255
    average = np.average(edges, axis=1)
    for i in range(shape[0]):
        if average[i] > threshold:
            edges[i,:] = 0
    average = np.average(edges, axis=0)
    for i in range(shape[1]):
        if average[i] > threshold:
            edges[:,i] = 0

def MergeFiles(files, bufsize=1024*1024):
    newStem = Path(files[0]).stem.replace('_trimmed', '') + '_merged_trimmed'
    
    # merge .ptsmap .markermap
    ptsmapMerged = {}
    markerMapMerged = {}
    for path in files:
        path = Path(path)
        ptsmap, markerMap = LoadExistingData(path.parent / '_metadata' / (path.stem + '.ptsmap'), path.parent / '_metadata' / (path.stem + '.markermap'))
        if ptsmapMerged == {} and markerMapMerged == {}:
            ptsmapMerged, markerMapMerged = ptsmap, markerMap
        else:
            k, v = ptsmapMerged.popitem()
            currentDuration = v['prev_end_pts']
            currentLen = v['prev_end_pos']
            del ptsmap['0.0']
            # append ptsmap
            for k, v in ptsmap.items():
                pts_merged = float(k) + currentDuration
                ptsmapMerged[str(pts_merged)] = {
                    'pts_display': FormatTimestamp(pts_merged),
                    'sad': v['sad'],
                    #'silent_ss': v['silent_ss'] + currentDuration,
                    #'silent_to': v['silent_to'] + currentDuration,
                    'prev_end_pts': v['prev_end_pts'] + currentDuration,
                    'prev_end_sad': v['prev_end_sad'],
                    'prev_end_pos': v['prev_end_pos'] + currentLen,
                    'next_start_pts': v['next_start_pts'] + currentDuration,
                    'next_start_sad': v['next_start_sad'],
                    'next_start_pos': v['next_start_pos'] + currentLen,
                }
            # append markermap
            for k, v in markerMap.items():
                clip = eval(k)
                clipMerged = (clip[0] + currentDuration, clip[1] + currentDuration)
                markerMapMerged[str(clipMerged)] = markerMap[k]
    newIndexFile = path.parent / '_metadata' / (newStem + '.ptsmap')
    newMarkerPath = path.parent / '_metadata' / (newStem + '.markermap')
    with newIndexFile.open('w') as f1, newMarkerPath.open('w') as f2:
        json.dump(ptsmapMerged, f1, indent=True)
        json.dump(markerMapMerged, f2, indent=True)
    
    # modify .toencode file
    for suffix in ('.toencode', '.error'):
        taskPath = Path(files[0].replace('_trimmed', '')).with_suffix(suffix)
        if taskPath.exists():
            with taskPath.open(encoding='utf-8') as rf:
                task = json.load(rf)
                task['path'] = str(Path(task['path']).with_stem(newStem)).replace('_trimmed', '')
                newTaskPath = taskPath.with_stem(newStem)
                with newTaskPath.open('w', encoding='utf-8') as wf:
                    json.dump(task, wf, ensure_ascii=False, indent=True)
            break

    # merge mpegts files
    totalSize = list(ptsmapMerged.values())[-1]['prev_end_pos']
    newTsFile = path.with_stem(newStem)
    if newTsFile.exists():
        newTsFile.unlink()
    with tqdm(total=totalSize, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        with newTsFile.open('ab') as wf:
            for path in files:
                with Path(path).open('rb') as rf:
                    data = rf.read(bufsize)
                    while len(data) > 0:
                        wf.write(data)
                        pbar.update(len(data))
                        data = rf.read(bufsize)
    
    # cut CMs
    CutCMs(videoPath=newTsFile, indexPath=ptsmapMerged, markerPath=markerMapMerged, byMethod='_groundtruth', outputFolder=newTsFile.parent / newTsFile.stem)
                
