from functools import lru_cache
import logging, re, json
from pathlib import Path
from tqdm import tqdm
import pysubs2

logger = logging.getLogger('tsmarker.dataset')

@lru_cache(maxsize=2)
def ExtractSubs(assPath: Path) -> "pysubs2.SSAFile":
    return pysubs2.load(str(assPath))

def ExtractSubtitlesText(assPath: Path, clip: tuple[float, float]) -> str:
    try:
        result = []
        for event in ExtractSubs(assPath): # type: ignore
            start, end = clip[0] * 1000, clip[1] * 1000
            # if overlapped
            if event.start < end and start < event.end:
                text = event.text
                text = re.sub(r'\{.*?\}', '', text)
                text = text.replace(r'\N', '')
                result.append(text)
        return ' '.join(result)
    except FileNotFoundError:
        return ""

def ExtractGenSubtitlesText(assGenPath: Path, clip: tuple[float, float]) -> str:
    try:
        with assGenPath.open() as f:
            assGen = json.load(f)
        return assGen[str(clip)]
    except KeyError:
        return ""
    except FileNotFoundError:
        return ""

