import pysubs2
import tsutils.subtitles
from tsmarker.common import LoadExistingData, GetClips, SaveMarkerMap

def Overlap(range1, range2):
    return (range1[0] <= range2[0] <= range1[1]) or (range2[0] <= range1[0] <= range2[1])

def Mark(videoPath, indexPath, markerPath):
    ptsMap, markerMap = LoadExistingData(indexPath=indexPath, markerPath=markerPath)
    tsutils.subtitles.Extract(videoPath)
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
