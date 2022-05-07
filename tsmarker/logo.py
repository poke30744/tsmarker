import argparse, tempfile, shutil, os, subprocess
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2 as cv
from PIL import Image
import tscutter.ffmpeg
from tscutter.common import ClipToFilename, InvalidTsFormat
from tscutter.analyze import SplitVideo
from .common import LoadExistingData, GetClips, SaveMarkerMap, SelectClips, RemoveBoarder

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

class InputFile(tscutter.ffmpeg.InputFile):
    def ExtractArea(self, area, folder, ss, to, fps='1/1', quiet=False):
        folder = self.path.with_suffix('') if folder is None else Path(folder)
        if folder.is_dir():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

        info = self.GetInfo()
        w, h, x, y = int(round(area[2] * info['width'])), int(round(area[3] * info['height'])), int(round(area[0] * info['width'])), int(round(area[1] * info['height']))
        args = [ 'ffmpeg', '-hide_banner' ]
        if ss is not None and to is not None:
            args += [ '-ss', str(ss), '-to', str(to) ]
        fpsStr = ',fps={}'.format(fps) if fps else ''
        args += [
            '-i', self.path,
            '-filter:v', 'crop={}:{}:{}:{}{}'.format(w, h, x, y, fpsStr),
            '{}/out%8d.bmp'.format(folder) ]
        pipeObj = subprocess.Popen(args, stderr=subprocess.PIPE, universal_newlines='\r', errors='ignore')
        if to > info['duration']:
            to = info['duration']
        with tqdm(total=to - ss, disable=quiet, unit='secs') as pbar:
            pbar.set_description('Extracting area')
            for line in pipeObj.stderr:
                if 'time=' in line:
                    for item in line.split(' '):
                        if item.startswith('time='):
                            timeFields = item.replace('time=', '').split(':')
                            time = float(timeFields[0]) * 3600 + float(timeFields[1]) * 60  + float(timeFields[2])
                            pbar.update(time - pbar.n)
            pbar.update(to - ss - pbar.n)
        pipeObj.wait()

    def ExtractLogo(self, area, outputPath, ss=0, to=999999, fps='1/1', quiet=False):
        outputPath = self.path.parent / (self.path.stem + '_logo.png') if outputPath is None else Path(outputPath)
        with tempfile.TemporaryDirectory(prefix='logo_pics_') as tmpLogoFolder:
            self.ExtractArea(area=area, folder=tmpLogoFolder, ss=ss, to=to, fps=fps, quiet=quiet)
            pics = list(Path(tmpLogoFolder).glob('*.bmp'))
            picSum = None
            for path in tqdm(pics, desc='Loading pics', total=len(pics), disable=quiet):
                image = np.array(Image.open(path)).astype(np.float32)
                picSum = image if picSum is None else (picSum + image)
            if picSum is None:
                raise InvalidTsFormat(f'"{self.path.name}" is invalid!')
        picSum /= len(pics)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(picSum.astype(np.uint8)).save(str(outputPath))
        return outputPath

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

def Mark(videoPath, indexPath, markerPath, quiet=False):
    videoPath = Path(videoPath)
    ptsMap, markerMap = LoadExistingData(indexPath=indexPath, markerPath=markerPath)
    clips = GetClips(ptsMap)
    
    # extract clip logos
    videoFolder = SplitVideo(videoPath=videoPath, indexPath=indexPath)
    for clip in tqdm(clips, desc='extracing logo edges'):
        clipLen = clip[1] - clip[0]
        logoPath = videoFolder / Path(ClipToFilename(clip)).with_suffix('.png')
        try:
            inputFile = InputFile(videoFolder / ClipToFilename(clip))
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
    selectedClips, selectedLen = SelectClips(clips)
    if selectedLen == 0:
        selectedClips, selectedLen = SelectClips(clips, lengthLimit=15)
    if selectedLen == 0:
        selectedClips, selectedLen = SelectClips(clips, lengthLimit=0)
    for clip in selectedClips:
        clipLen = clip[1] - clip[0]
        logoPath = videoFolder / Path(ClipToFilename(clip)).with_suffix('.png')
        edgePath = drawEdges(logoPath, removeBoarder=True)
        img = cv2imread(str(logoPath)) * clipLen
        videoLogo = img if videoLogo is None else (videoLogo + img)
    videoLogo /= selectedLen
    logoPath = videoFolder.parent / (videoPath.stem + '_logo.png')
    cv2imwrite(str(logoPath), videoLogo)
    videoEdgePath = drawEdges(logoPath, removeBoarder=True)

    # mark
    videoEdge = cv2imread(str(videoEdgePath), 0)
    for clip in clips:
        edgePath = videoFolder / Path(ClipToFilename(clip)).with_suffix('.edge.png')
        clipEdge = cv2imread(str(edgePath), 0)
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
        InputFile(args.input).ExtractLogo(area=args.area, outputPath=args.output, ss=args.ss, to=args.to, fps=args.fps)