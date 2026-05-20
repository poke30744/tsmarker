import logging, subprocess, tempfile
from pathlib import Path
import pysubs2
from . import common

logger = logging.getLogger('tsmarker.subtitles')


def Overlap(range1, range2):
    return (range1[0] <= range2[0] <= range1[1]) or (range2[0] <= range1[0] <= range2[1])


class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, assPath: Path = None, progress=None) -> None:
        subtitles = None

        # Extract embedded ASS from MKV
        with tempfile.NamedTemporaryFile(suffix='.ass', delete=False) as tmp:
            tmpPath = Path(tmp.name)
        try:
            subprocess.run(['ffmpeg', '-y', '-hide_banner',
                            '-i', str(videoPath), '-map', '0:s:0', '-c:s', 'copy',
                            str(tmpPath)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if tmpPath.stat().st_size > 0:
                subtitles = pysubs2.load(tmpPath, encoding='utf-8')
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        finally:
            tmpPath.unlink(missing_ok=True)

        clips = self.Clips()
        if subtitles is not None:
            tid = "subtitles_mark"
            if progress is not None:
                progress.add_task(tid, len(clips), "Marking subtitles")
            for i, clip in enumerate(clips):
                overlap = False
                for event in subtitles:
                    if not event.text.strip():
                        continue
                    if Overlap((event.start / 1000, event.end / 1000), (clip[0], clip[1])):
                        overlap = True
                        break
                self.Mark(clip, 'subtitles', 1.0 if overlap else 0.0)
                if progress is not None:
                    progress.update(tid, i + 1)
            if progress is not None:
                progress.done(tid)
        else:
            logger.warning(f'No ASS file found at {assPath}, marking all clips as 0.5')
            for clip in clips:
                self.Mark(clip, 'subtitles', 0.5)
        self.Save()
