from tqdm import tqdm
from tscutter.ffmpeg import InputFile
from .common import LoadExistingData, GetClips, SaveMarkerMap

def Mark(videoPath, indexPath, markerPath, quiet=False):
    ptsMap, markerMap = LoadExistingData(indexPath=indexPath, markerPath=markerPath)
    clips = GetClips(ptsMap)
    videoInfo = InputFile(videoPath).GetInfo()
    videoDuration = videoInfo['duration']
    for i in tqdm(range(len(clips)), desc='Marking clip info'):
        clip = clips[i]
        markerMap[str(clip)]['position'] = clip[0] / videoDuration
        markerMap[str(clip)]['duration'] = clip[1] - clip[0]
    for i in range(len(clips)):
        clip = clips[i]
        markerMap[str(clip)]['duration_prev'] = 0.0 if i == 0 else markerMap[str(clips[i - 1])]['duration']
        markerMap[str(clip)]['duration_next'] = 0.0 if i == len(clips) - 1 else markerMap[str(clips[i + 1])]['duration']
    markerPath = SaveMarkerMap(markerMap, markerPath)
    return markerPath