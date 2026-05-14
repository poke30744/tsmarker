import tempfile
from pathlib import Path
import math
import numpy as np
import cv2 as cv
from .ptsmap import ClipToFilename
from .inputfile import InvalidTsFormat
from . import common
from .pipeline import ExtractLogoPipeline, cv2imread, InputFile


def _detect_logo_region(mean_img):
    """Find the logo region in a mean image by scanning for highest edge density."""
    h, w = mean_img.shape
    edges = cv.Canny(mean_img.astype(np.uint8), 20, 50, 3)
    best_score, best_rect = 0, (0, 0, 30, 80)
    for y in range(0, min(200, h), 4):
        for x in range(w // 2, w, 10):
            for hh in [30, 50, 70, 90]:
                for ww in [80, 120, 160, 200, 250, 300]:
                    if x + ww > w or y + hh > h:
                        continue
                    patch = edges[y:y + hh, x:x + ww]
                    density = np.count_nonzero(patch) / patch.size
                    score = density / (hh * ww) ** 0.25
                    if score > best_score:
                        best_score = score
                        best_rect = (y, x, hh, ww)
    return best_rect


def _ncc(template_vals, test_vals):
    """Normalized cross-correlation between two flattened pixel arrays."""
    t = template_vals.astype(np.float64)
    i = test_vals.astype(np.float64)
    t = t - t.mean()
    i = i - i.mean()
    n = np.dot(t, i)
    d = np.sqrt(np.dot(t, t) * np.dot(i, i))
    return float(n / d) if d > 0 else 0.0


class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, logoPath: Path = None, maxTimeToExtract=60, progress=None) -> None:
        with tempfile.TemporaryDirectory(prefix='logo_MarkerMap_MarkAll_') as tmpFolder:
            if logoPath is None or not logoPath.exists():
                logoPath = Path(tmpFolder) / videoPath.with_suffix('.logo.png').name
                ExtractLogoPipeline(inFile=videoPath, ptsMap=self.ptsMap, outFile=logoPath, maxTimeToExtract=999999)
                logoMean = cv2imread(logoPath, 0)
                logoPath.unlink()
            else:
                logoMean = cv2imread(logoPath, 0)

            if logoMean is None:
                return

            ry, rx, rh, rw = _detect_logo_region(logoMean)

            clips = self.Clips()
            tid = "detect_logo"
            if progress is not None:
                progress.add_task(tid, len(clips), "Detecting logo")
            for i, clip in enumerate(clips):
                logoScore = self.ExtractLogoScore(videoPath, clip, maxTimeToExtract,
                                                  Path(tmpFolder), logoMean, ry, rx, rh, rw)
                if logoScore <= 0.5:
                    logoScore = self.ExtractLogoScore(videoPath, clip, 999999,
                                                      Path(tmpFolder), logoMean, ry, rx, rh, rw)
                self.Mark(clip, 'logo', logoScore)
                if progress is not None:
                    progress.update(tid, i + 1)
            if progress is not None:
                progress.done(tid)
        self.Save()

    def ExtractLogoScore(self, videoPath: Path, clip: list, maxTimeToExtract: float,
                         tmpFolder: Path, logoMean, ry, rx, rh, rw) -> float:
        if clip[1] - clip[0] > maxTimeToExtract:
            padding = (clip[1] - clip[0] - maxTimeToExtract) / 2
            realClip = (padding + clip[0], padding + clip[0] + maxTimeToExtract)
        else:
            realClip = clip
        clipMeanImagePath = tmpFolder / Path(ClipToFilename(clip)).with_suffix('.png')
        try:
            inputFile = InputFile(videoPath)
            inputFile.ExtractMeanImage(clip=realClip, outFile=clipMeanImagePath)
        except InvalidTsFormat:
            return 0.0

        clipMean = cv2imread(clipMeanImagePath, 0)
        if clipMean is None or clipMean.shape != logoMean.shape:
            return 0.0

        tpl_vals = logoMean[ry:ry + rh, rx:rx + rw].flatten()
        clip_vals = clipMean[ry:ry + rh, rx:rx + rw].flatten()
        score = _ncc(tpl_vals, clip_vals)
        if math.isnan(score):
            return 0.0
        return max(0.0, score)
