import json
import numpy as np

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
    if markerPath.exists():
        with markerPath.open() as f:
            existingMarkerMap = json.load(f)
    else:
        existingMarkerMap = None
    if existingMarkerMap != markerMap:
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