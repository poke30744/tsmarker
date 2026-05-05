import tempfile
from pathlib import Path
import math
import numpy as np
from tscutter.common import ClipToFilename, InvalidTsFormat
from . import common
from .pipeline import ExtractLogoPipeline, cv2imread, drawEdges, InputFile

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, logoPath: Path=None, maxTimeToExtract=10, progress=None) -> None:
        with tempfile.TemporaryDirectory(prefix='logo_MarkerMap_MarkAll_') as tmpFolder:
            if logoPath is None or not logoPath.exists():
                logoPath = Path(tmpFolder) / videoPath.with_suffix('.logo.png').name
                ExtractLogoPipeline(inFile=videoPath, ptsMap=self.ptsMap, outFile=logoPath, maxTimeToExtract=999999)
                logoEdge = cv2imread(logoPath, 0)
                logoPath.unlink()
            else:
                logoEdge = cv2imread(logoPath, 0)

            clips = self.Clips()
            tid = "detect_logo"
            if progress is not None:
                progress.add_task(tid, len(clips), "Detecting logo")
            for i, clip in enumerate(clips):
                logoScore = self.ExtractLogoScore(videoPath, clip, maxTimeToExtract, tmpFolder, logoEdge)
                if logoScore <= 0.5:
                    logoScore = self.ExtractLogoScore(videoPath, clip, 999999, tmpFolder, logoEdge)
                self.Mark(clip, 'logo', logoScore)
                if progress is not None:
                    progress.update(tid, i + 1)
            if progress is not None:
                progress.done(tid)
        self.Save()

    def ExtractLogoScore(self, videoPath: Path, clip: list, maxTimeToExtract: float, tmpFolder: str, logoEdge) -> float:
        if clip[1] - clip[0] > maxTimeToExtract:
            padding = (clip[1] - clip[0] - maxTimeToExtract) / 2
            realClip = (padding + clip[0], padding + clip[0] + maxTimeToExtract)
        else:
            realClip = clip
        clipMeanImagePath = Path(tmpFolder) / Path(ClipToFilename(clip)).with_suffix('.png')
        try:
            inputFile = InputFile(videoPath)
            inputFile.ExtractMeanImagePipe(ptsMap=self.ptsMap, clip=realClip, outFile=clipMeanImagePath, progress=None)
        except InvalidTsFormat:
            return 0
        
        clipEdgePath = drawEdges(clipMeanImagePath)
        clipEdge = cv2imread(clipEdgePath, 0)
        if logoEdge.shape != clipEdge.shape:
            return 0
        andImage = np.bitwise_and(logoEdge, clipEdge)
        logoScore = np.sum(andImage) / np.sum(logoEdge)
        if math.isnan(logoScore):
            return 0
        return logoScore