import argparse, shutil, os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2 as cv
from PIL import Image
from tscutter.common import ClipToFilename, InvalidTsFormat
from . import common
from .ffmpeg import InputFile

# Unicode workaround of Python OpenCV from https://qiita.com/SKYS/items/cbde3775e2143cad7455
def cv2imread(filename, flags=cv.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def cv2imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

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

def drawEdges(imagePath, outputPath=None, threshold1=32, threshold2=64, apertureSize=3, removeBoarder=False):
    imagePath = Path(imagePath)
    outputPath = imagePath.with_suffix('.edge.png') if outputPath is None else Path(outputPath)
    img = cv2imread(str(imagePath), 0)
    edges = cv.Canny(img, threshold1, threshold2, apertureSize=3)
    if removeBoarder:
        # To remove boarder-like edges (like 4:3 picture in 16:9) from logos
        RemoveBoarder(edges)
    cv2imwrite(str(outputPath), edges)
    return outputPath

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, quiet=False) -> None:
        # extract clip logos
        outputFolder = videoPath.with_suffix('')
        self.ptsMap.SplitVideo(videoPath=videoPath, outputFolder=outputFolder, quiet=quiet)
        clips = self.Clips()
        for clip in tqdm(clips, desc='extracing logo edges', disable=quiet):
            clipLen = clip[1] - clip[0]
            logoPath = outputFolder / Path(ClipToFilename(clip)).with_suffix('.png')
            try:
                inputFile = InputFile(outputFolder / ClipToFilename(clip))
                inputFile.ExtractLogo(
                    area=[0.0, 0.0, 1.0, 1.0],
                    outputPath=logoPath,
                    to=clipLen if clipLen < 600 else 600, 
                    quiet=True)
            except InvalidTsFormat:
                # create a blank PNG as the place holder
                info = InputFile(videoPath).GetInfo()
                img = Image.new("RGB", (info['width'], info['height']), (0, 0, 0))
                img.save(logoPath, "PNG")
            edgePath = drawEdges(logoPath, removeBoarder=True)
            img = cv2imread(str(logoPath)) * clipLen
        
        # calculate the logo of the entire video
        videoLogo = None
        selectedClips, selectedLen = self.ptsMap.SelectClips()
        if selectedLen == 0:
            selectedClips, selectedLen = self.ptsMap.SelectClips(lengthLimit=15)
        if selectedLen == 0:
            selectedClips, selectedLen = self.ptsMap.SelectClips(lengthLimit=0)
        for clip in selectedClips:
            clipLen = clip[1] - clip[0]
            logoPath = outputFolder / Path(ClipToFilename(clip)).with_suffix('.png')
            edgePath = drawEdges(logoPath, removeBoarder=True)
            img = cv2imread(str(logoPath)) * clipLen
            videoLogo = img if videoLogo is None else (videoLogo + img)
        videoLogo /= selectedLen
        logoPath = outputFolder.parent / (videoPath.stem + '_logo.png')
        cv2imwrite(str(logoPath), videoLogo)
        videoEdgePath = drawEdges(logoPath, removeBoarder=True)

        # mark
        videoEdge = cv2imread(str(videoEdgePath), 0)
        for clip in clips:
            edgePath = outputFolder / Path(ClipToFilename(clip)).with_suffix('.edge.png')
            clipEdge = cv2imread(str(edgePath), 0)
            andImage = np.bitwise_and(videoEdge, clipEdge)
            logoScore = np.sum(andImage) / np.sum(videoEdge)
            self.Mark(clip, 'logo', logoScore)
        
        # cleanup
        shutil.rmtree(outputFolder)

        self.Save()

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
        InputFile(args.input).ExtractLogo(area=args.area, outputPath=args.output, ss=args.ss, to=args.to, fps=args.fps)