import tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from tscutter.common import ClipToFilename, InvalidTsFormat
from . import common
from .pipeline import ExtractLogoPipeline, cv2imread, drawEdges

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, logoPath: Path=None, maxTimeToExtract=10, quiet=False) -> None:
        with tempfile.TemporaryDirectory(prefix='logo_MarkerMap_MarkAll_') as tmpFolder:
            if logoPath is None or not logoPath.exists():
                # extract logo from the video
                logoPath = Path(tmpFolder) / videoPath.with_suffix('.logo.png').name
                ExtractLogoPipeline(inFile=videoPath, ptsMap=self.ptsMap, outFile=logoPath)
                logoEdge = cv2imread(logoPath, 0)
                logoPath.unlink()
            else:
                logoEdge = cv2imread(logoPath, 0)
                
            clips = self.Clips()
            for clip in tqdm(clips, desc='Detecting logo ...', disable=quiet):
                if clip[1] - clip[0] > maxTimeToExtract:
                    padding = (clip[1] - clip[0] - maxTimeToExtract) / 2
                    realClip = (padding + clip[0], padding + clip[0] + maxTimeToExtract)
                else:
                    realClip = clip
                clipMeanImagePath = Path(tmpFolder) / Path(ClipToFilename(clip)).with_suffix('.png')
                try:
                    self.ptsMap.ExtractMeanImagePipe(inFile=videoPath, clip=realClip, outFile=clipMeanImagePath, quiet=True)
                except InvalidTsFormat:
                    self.Mark(clip, 'logo', 0)
                    continue
                clipEdgePath = drawEdges(clipMeanImagePath)
                # mark
                clipEdge = cv2imread(clipEdgePath, 0)
                andImage = np.bitwise_and(logoEdge, clipEdge)
                logoScore = np.sum(andImage) / np.sum(logoEdge)
                self.Mark(clip, 'logo', logoScore)
        
        self.Save()