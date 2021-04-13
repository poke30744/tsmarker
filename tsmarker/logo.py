import argparse, tempfile, shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2 as cv
from PIL import Image
from tsutils.ffmpeg import ExtractArea, GetInfo
from tsutils.common import ClipToFilename, InvalidTsFormat
from tscutter.analyze import SplitVideo
from .common import LoadExistingData, GetClips, SaveMarkerMap, SelectClips, RemoveBoarder

def ExtractLogo(videoPath, area, outputPath, ss=0, to=999999, fps='1/1', quiet=False):
    videoPath = Path(videoPath)
    outputPath = videoPath.parent / (videoPath.stem + '_logo.png') if outputPath is None else Path(outputPath)
    with tempfile.TemporaryDirectory(prefix='logo_pics_') as tmpLogoFolder:
        ExtractArea(path=videoPath, area=(0.0, 0.0, 1.0, 1.0), folder=tmpLogoFolder, ss=ss, to=to, fps=fps, quiet=quiet)
        pics = list(Path(tmpLogoFolder).glob('*.bmp'))
        picSum = None
        for path in tqdm(pics, desc='Loading pics', total=len(pics), disable=quiet):
            image = np.array(Image.open(path)).astype(np.float32)
            picSum = image if picSum is None else (picSum + image)
        if picSum is None:
            raise InvalidTsFormat(f'"{videoPath.name}" is invalid!')
    picSum /= len(pics)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(picSum.astype(np.uint8)).save(str(outputPath))
    return outputPath

def drawEdges(imagePath, outputPath=None, threshold1=32, threshold2=64, apertureSize=3):
    imagePath = Path(imagePath)
    outputPath = imagePath.with_suffix('.edge.png') if outputPath is None else Path(outputPath)
    img = cv.imread(str(imagePath), 0)
    edges = cv.Canny(img, threshold1, threshold2, apertureSize=3)
    # To remove board-like edges (like 4:3 picture in 16:9) from logos
    RemoveBoarder(edges)
    cv.imwrite(str(outputPath), edges)
    return outputPath

def Mark(videoPath, indexPath, markerPath, quiet=False):
    videoPath = Path(videoPath)
    ptsMap, markerMap = LoadExistingData(indexPath=indexPath, markerPath=markerPath)
    clips = GetClips(ptsMap)
    
    # extract clip logos
    videoFolder = SplitVideo(videoPath=videoPath)
    for clip in tqdm(clips, desc='extracing logo edges'):
        clipLen = clip[1] - clip[0]
        logoPath = videoFolder / Path(ClipToFilename(clip)).with_suffix('.png')
        try:
            ExtractLogo(
                videoPath=videoFolder / ClipToFilename(clip),
                area=[0.0, 0.0, 1.0, 1.0],
                outputPath=logoPath,
                to=clipLen if clipLen < 600 else 600, 
                quiet=True)
        except InvalidTsFormat:
            # create a blank PNG as the place holder
            info = GetInfo(videoPath)
            img = Image.new("RGB", (info['width'], info['height']), (0, 0, 0))
            img.save(logoPath, "PNG")
        edgePath = drawEdges(logoPath)
        img = cv.imread(str(logoPath)) * clipLen
    
    # calculate the logo of the entire video
    videoLogo = None
    selectedClips, selectedLen = SelectClips(clips)
    if selectedLen == 0:
        selectedClips, selectedLen = SelectClips(clips, lengthLimit=15)
    if selectedLen == 0:
        selectedClips, selectedLen = SelectClips(clips, lengthLimit=0)
    for clip in selectedClips:
        clipLen = clip[1] - clip[0]
        logoPath = videoFolder / Path(ClipToFilename(clip)).with_suffix('.png')
        edgePath = drawEdges(logoPath)
        img = cv.imread(str(logoPath)) * clipLen
        videoLogo = img if videoLogo is None else (videoLogo + img)
    videoLogo /= selectedLen
    logoPath = videoFolder.parent / (videoPath.stem + '_logo.png')
    cv.imwrite(str(logoPath), videoLogo)
    videoEdgePath = drawEdges(logoPath)

    # mark
    videoEdge = cv.imread(str(videoEdgePath), 0)
    for clip in clips:
        edgePath = videoFolder / Path(ClipToFilename(clip)).with_suffix('.edge.png')
        clipEdge = cv.imread(str(edgePath), 0)
        andImage = np.bitwise_and(videoEdge, clipEdge)
        logoScore = np.sum(andImage) / np.sum(videoEdge)
        markerMap[str(clip)]['logo'] = logoScore
    
    # cleanup
    shutil.rmtree(videoFolder)

    markerPath = SaveMarkerMap(markerMap, markerPath)
    return markerPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logo tools')
    parser.add_argument('--quiet', '-q', action='store_true', help="don't output to the console")
    subparsers = parser.add_subparsers(required=True, title='subcommands', dest='command')

    subparser = subparsers.add_parser('logo', help='Extract logo from the video')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')
    subparser.add_argument('--area', default=[0.0, 0.0, 1.0, 1.0], nargs=4, type=float, help='the area to extract')
    subparser.add_argument('--ss', type=float, default=0, help='from (seconds)')
    subparser.add_argument('--to', type=float, default=999999, help='to (seconds)')
    subparser.add_argument('--fps', default='1/1', help='fps like 1/1')
    subparser.add_argument('--output', '-o', help='output image path')

    args = parser.parse_args()

    if args.command == 'logo':
        ExtractLogo(videoPath=args.input, area=args.area, outputPath=args.output, ss=args.ss, to=args.to, fps=args.fps)