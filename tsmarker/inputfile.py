import json, shutil
from dataclasses import dataclass
from pathlib import Path
import ffmpeg


class InvalidTsFormat(RuntimeError): ...


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int


class InputFile:
    def __init__(self, path) -> None:
        self.ffmpeg = shutil.which('ffmpeg')
        self.ffprobe = shutil.which('ffprobe')
        if self.ffmpeg is None:
            raise RuntimeError("ffmpeg not found in PATH — install ffmpeg or add it to PATH")
        if self.ffprobe is None:
            raise RuntimeError("ffprobe not found in PATH — install ffmpeg or add it to PATH")
        self.path = Path(path)

    def GetInfo(self) -> VideoInfo:
        try:
            probeInfo = ffmpeg.probe(str(self.path), cmd=self.ffprobe, show_programs=None)
        except (ffmpeg.Error, json.JSONDecodeError, KeyError):
            raise InvalidTsFormat(f'"{self.path.name}" is invalid!')
        video_stream = next(s for s in probeInfo['streams'] if s.get('codec_type') == 'video')
        duration = float(video_stream.get('duration') or probeInfo['format']['duration'])
        return VideoInfo(
            duration=duration,
            width=video_stream['width'],
            height=video_stream['height'],
        )
