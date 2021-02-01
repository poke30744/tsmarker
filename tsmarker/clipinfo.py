#!/usr/bin/env python3
import argparse, tempfile
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
from tsmarker.common import LoadExistingData, GetClips, SaveMarkerMap
from tsutils.ffmpeg import GetInfo, ExtractArea
from tsutils.encode import FindVideoBox

def Mark(videoPath, indexPath, markerPath, quiet=False):
    ptsMap, markerMap = LoadExistingData(indexPath=indexPath, markerPath=markerPath)
    clips = GetClips(ptsMap)
    videoInfo = GetInfo(videoPath)
    #sar = videoInfo['sar']
    videoDuration = videoInfo['duration']
    for i in tqdm(range(len(clips)), desc='Marking clip info'):
        clip = clips[i]
        markerMap[str(clip)]['position'] = clip[0] / videoDuration
        markerMap[str(clip)]['duration'] = clip[1] - clip[0]
        #_, __, w, h = FindVideoBox(path=videoPath, ss=clip[0], to=clip[1], quiet=True)
        #markerMap[str(clip)]['aspect_ratio'] = round((w * sar[0]) / (h * sar[1]), 2)
    for i in range(len(clips)):
        clip = clips[i]
        markerMap[str(clip)]['duration_prev'] = 0.0 if i == 0 else markerMap[str(clips[i - 1])]['duration']
        markerMap[str(clip)]['duration_next'] = 0.0 if i == len(clips) - 1 else markerMap[str(clips[i + 1])]['duration']
        #markerMap[str(clip)]['aspect_ratio_prev'] = 0.0 if i == 0 else markerMap[str(clips[i - 1])]['aspect_ratio']
        #markerMap[str(clip)]['aspect_ratio_next'] = 0.0 if i == len(clips) - 1 else markerMap[str(clips[i + 1])]['aspect_ratio']
    markerPath = SaveMarkerMap(markerMap, markerPath)
    return markerPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python tool to retrieve video clip info')
    parser.add_argument('--input', '-i', required=True, help='input mpegts path')
    parser.add_argument('--quiet', '-q', action='store_true', help="don't output to the console")

    args = parser.parse_args()

    info = GetInfo(args.input)
    print(f'info: {info}')

    videoBox = FindVideoBox(args.input)
    x, y, w, h = videoBox
    aspectRatio = round((w * info['sar'][0]) / (h * info['sar'][1]), 2)
    percentage = round(w * h / (info['width'] * info['height']) * 100, 2)
    print(f'videoBox: {videoBox} ({percentage}%), aspectRatio: {aspectRatio}')