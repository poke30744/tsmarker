import tempfile
from pathlib import Path
import math
from tqdm import tqdm
import numpy as np
from tscutter.common import ClipToFilename, InvalidTsFormat
from . import common
from .pipeline import ExtractLogoPipeline, cv2imread, drawEdges, InputFile

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, logoPath: Path=None, maxTimeToExtract=10, quiet=False) -> None:
        with tempfile.TemporaryDirectory(prefix='logo_MarkerMap_MarkAll_') as tmpFolder:
            if logoPath is None or not logoPath.exists():
                # extract logo from the video
                logoPath = Path(tmpFolder) / videoPath.with_suffix('.logo.png').name
                ExtractLogoPipeline(inFile=videoPath, ptsMap=self.ptsMap, outFile=logoPath, maxTimeToExtract=999999)
                logoEdge = cv2imread(logoPath, 0)
                logoPath.unlink()
            else:
                logoEdge = cv2imread(logoPath, 0)
                
            clips = self.Clips()
            for clip in tqdm(clips, desc='Detecting logo ...', disable=quiet):
                logoScore = self.ExtractLogoScore(videoPath, clip, maxTimeToExtract, tmpFolder, logoEdge)
                if logoScore <= 0.5:
                    # try again to extract the entire duration of the clip
                    logoScore = self.ExtractLogoScore(videoPath, clip, 999999, tmpFolder, logoEdge)
                self.Mark(clip, 'logo', logoScore)
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
            inputFile.ExtractMeanImagePipe(ptsMap=self.ptsMap, clip=realClip, outFile=clipMeanImagePath, quiet=True)
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