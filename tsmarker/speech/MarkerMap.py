from pathlib import Path
import tempfile, requests, json
from .. import common
from .dataset import ExtractSubtitlesText
from ..subtitles import Extract

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, apiUrl: str) -> None:
        clips = self.Clips()
        with tempfile.TemporaryDirectory(prefix='speech') as tmpFolder:
            subtitlesPathList = Extract(videoPath, Path(tmpFolder))
            assPath = None
            for path in subtitlesPathList:
                if path.suffix == '.ass':
                    assPath = path
                    break
            textList = [ ExtractSubtitlesText(assPath, clip) for clip in clips ] if assPath is not None else [ '' ] * len(clips)
        payload = json.dumps(textList)
        response = requests.request("POST", apiUrl, data=payload, headers={ 'Content-Type': 'application/json' })
        Y = response.json()
        for i in range(len(clips)):
            self.Mark(clips[i], 'speech', float(Y[i][0]))
        self.Save()