import json

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