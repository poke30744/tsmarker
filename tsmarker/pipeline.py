import tempfile, subprocess, io, logging, os, argparse
from pathlib import Path
from threading import Thread
import numpy as np
from PIL import Image
import cv2 as cv
from tscutter import ffmpeg
from tscutter.common import PtsMap, InvalidTsFormat, ClipToFilename

logger = logging.getLogger('tsmarker.pipeline')

# Unicode workaround of Python OpenCV from https://qiita.com/SKYS/items/cbde3775e2143cad7455
def cv2imread(filename, flags=cv.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(str(filename), dtype)
        img = cv.imdecode(n, flags)
        return img
    except Exception as e:
        logger.error(f'cv2imread failed: {e}')
        return None

def cv2imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv.imencode(ext, img, params)
        if result:
            with open(str(filename), mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        logger.error(f'cv2imwrite failed: {e}')
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
    img = cv2imread(imagePath, 0)
    edges = cv.Canny(img, threshold1, threshold2, apertureSize=3)
    if removeBoarder:
        # To remove boarder-like edges (like 4:3 picture in 16:9) from logos
        RemoveBoarder(edges)
    cv2imwrite(outputPath, edges)
    return outputPath

class InputFile(ffmpeg.InputFile):
    def ExtractAreaPipeCmd(self, inFile, folder, crop=None, ss=None, to=None, fps='1/1'):
        args = [ self.ffmpeg, '-hide_banner' ]
        if ss is not None and to is not None:
            args += [ '-ss', str(ss), '-to', str(to) ]
        args += [ '-i', str(inFile) ]
        vFilters = []
        if crop is not None:
            vFilters += [ f'crop={crop["w"]}:{crop["h"]}:{crop["x"]}:{crop["y"]}' ]
        if fps is not None:
            vFilters += [ f'fps={fps}' ]
        if vFilters:
            args += [ '-filter:v', ','.join(vFilters) ]
        args += [ f'{folder}/out%8d.bmp' ]
        return args

    def HandleFFmpegProgress(lines, callback=None, progress=None, tid=None):
        last_time = 0.0
        for line in lines:
            if 'time=' in line:
                for item in line.split(' '):
                    if item.startswith('time='):
                        timeFields = item.replace('time=', '').split(':')
                        try:
                            time = float(timeFields[0]) * 3600 + float(timeFields[1]) * 60 + float(timeFields[2])
                        except ValueError:
                            continue
                        if progress is not None and tid is not None:
                            progress.update(tid, time)
                        last_time = time
                        if callback is not None:
                            callback()
        if progress is not None and tid is not None:
            progress.done(tid)

    def HandleFFmpegLog(info, lines, callback=None, progress=None) -> None:
        tid = None
        if progress is not None and info.duration is not None:
            tid = "ffmpeg_logo"
            progress.add_task(tid, info.duration, "Extracting logo frames", unit="s")
        InputFile.HandleFFmpegProgress(lines, callback=callback, progress=progress, tid=tid)

    def ExtractMeanImagePipe(self, ptsMap: PtsMap, clip: tuple[float], outFile: Path, progress=None):
        info = self.GetInfo()
        with tempfile.TemporaryDirectory(prefix='LogoPipeline_') as tmpFolder:
            with subprocess.Popen(self.ExtractAreaPipeCmd('-', tmpFolder), stdin=subprocess.PIPE, stderr=subprocess.PIPE) as extractAreaP:
                thread = Thread(target=ptsMap.ExtractClipPipe, args=(self.path, clip, extractAreaP.stdin, progress))
                thread.start()

                class LogoGenerator:
                    def __init__(self):
                        self.picSum = None
                        self.count = 0
                    def Callback(self):
                        for path in Path(tmpFolder).glob('*.bmp'):
                            image = np.array(Image.open(path)).astype(np.float32)
                            self.picSum = image if self.picSum is None else (self.picSum + image)
                            self.count += 1
                            path.unlink()
                    def Save(self, path):
                        Image.fromarray((self.picSum/self.count).astype(np.uint8)).save(str(path))

                logoGenerator = LogoGenerator()
                try:
                    InputFile.HandleFFmpegLog(info=info, lines=io.TextIOWrapper(extractAreaP.stderr, errors='ignore'), callback=logoGenerator.Callback, progress=progress)
                except (IndexError, UnboundLocalError):
                    raise InvalidTsFormat(f'"{self.path.name}" is invalid!')
                if logoGenerator.count > 0:
                    logoGenerator.Save(outFile)
                else:
                    Image.new("RGB", (info.width, info.height), (0, 0, 0)).save(str(outFile))

                thread.join()

def ExtractLogoPipeline(inFile: Path, ptsMap: PtsMap, outFile: Path, maxTimeToExtract=120, removeBoarder: bool=True, progress=None) -> None:
    selectedClips, selectedLen = ptsMap.SelectClips()
    if selectedLen == 0:
        selectedClips, selectedLen = ptsMap.SelectClips(lengthLimit=15)
    if selectedLen == 0:
        selectedClips, selectedLen = ptsMap.SelectClips(lengthLimit=0)
    with tempfile.TemporaryDirectory(prefix='ExtractLogoPipeline_') as tmpFolder:
        clip = max(selectedClips, key=lambda clip: clip[1] - clip[0])
        logoPath = tmpFolder / Path(ClipToFilename(clip)).with_suffix('.png')
        if clip[1] - clip[0] > maxTimeToExtract:
            padding = (clip[1] - clip[0] - maxTimeToExtract) / 2
            clip = (padding + clip[0], padding + clip[0] + maxTimeToExtract)
        logger.info(f'Extracting logo from {inFile.name}: {clip} ...')
        inputFile = InputFile(inFile)
        inputFile.ExtractMeanImagePipe(ptsMap, clip, logoPath, progress=progress)

        drawEdges(logoPath, outputPath=outFile, removeBoarder=removeBoarder)

def CropDetectPipeline(videoEdgePath, threshold=0.3):
    videoEdges = np.array(Image.open(videoEdgePath))
    xAxis = videoEdges.mean(axis=0)
    yAxis = videoEdges.mean(axis=1)
    try:
        xBoarders = np.argwhere(xAxis > 255 * threshold).flatten()
        yBoarders = np.argwhere(yAxis > 255 * threshold).flatten()
        # detect boarder like below
        # ........||.............|..........
        # ......|.............|||..........
        x1, x2 = (xBoarders[0], xBoarders[-1]) if len(xBoarders) >= 2 else (0, len(xAxis) - 1)
        y1, y2 = (yBoarders[0], yBoarders[-1]) if len(yBoarders) >= 2 else (0, len(yAxis) - 1)
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        return { 'w': w, 'h': h, 'x': x1, 'y': y1 }
    except IndexError:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TS files in pipeline')
    subparsers = parser.add_subparsers(required=True, title='subcommands', dest='command')

    subparser = subparsers.add_parser('logo', help='extract logo from mpegts file')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')

    subparser = subparsers.add_parser('cropdetect', help='detect crop parameters for mpegts file')
    subparser.add_argument('--input', '-i', required=True, help='input mpegts path')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.command == 'logo':
        inFile = Path(args.input)
        outFile = inFile.with_suffix('.logo.png')
        indexPath = inFile.parent / '_metadata' / (inFile.stem + '.ptsmap')
        ExtractLogoPipeline(inFile, PtsMap(indexPath), outFile)
    elif args.command == 'cropdetect':
        inFile = Path(args.input)
        indexPath = inFile.parent / '_metadata' / (inFile.stem + '.ptsmap')
        logoPath = inFile.parent / (inFile.stem + '_CropLogo.png')
        ExtractLogoPipeline(inFile, PtsMap(indexPath), logoPath, maxTimeToExtract=10, removeBoarder=False)
        cropInfo = CropDetectPipeline(logoPath)
        print(cropInfo)