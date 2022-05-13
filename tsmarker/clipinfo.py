from pathlib import Path
from tqdm import tqdm
from tscutter.ffmpeg import InputFile
from . import common

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, quiet=False) -> None:
        clips = self.Clips()
        videoInfo = InputFile(videoPath).GetInfo()
        videoDuration = videoInfo['duration']
        for i in tqdm(range(len(clips)), desc='Marking clip info', disable=quiet):
            clip = clips[i]
            self.Mark(clip, 'position', clip[0] / videoDuration)
            self.Mark(clip, 'duration', clip[1] - clip[0])
        for i in range(len(clips)):
            clip = clips[i]
            self.Mark(clip, 'duration_prev', 0.0 if i == 0 else self.Value(clips[i - 1], 'duration'))
            self.Mark(clip, 'duration_next', 0.0 if i == (len(clips) - 1) else self.Value(clips[i + 1], 'duration'))
        self.Save()