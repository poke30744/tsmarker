import logging
from pathlib import Path
from . import common

logger = logging.getLogger('tsmarker.groundtruth')

class GroundTruthError(RuntimeError): ...

class MarkerMap(common.MarkerMap):
    def MarkAll(self, clipsFolder: Path) -> bool:
        properties = self.Properties()
        cmFolder = clipsFolder / 'CM'
        dirty = False
        for clip in self.Clips():
            clipFilename = self.ClipToFilenameForReview(clip)
            if (clipsFolder / clipFilename).exists():
                groundTruth = 1.0
            elif (cmFolder / clipFilename).exists():
                groundTruth = 0.0
            else:
                raise GroundTruthError(f'{clip} does not exist in {clipsFolder}!')
            if '_groundtruth' in properties:
                if self.Value(clip, '_groundtruth') != groundTruth:
                    dirty = True
            elif '_ensemble' in properties:
                if self.Value(clip, '_ensemble') != groundTruth:
                    dirty = True
            else:
                dirty = True
            self.Mark(clip, '_groundtruth', groundTruth)
        self.Save()
        return dirty