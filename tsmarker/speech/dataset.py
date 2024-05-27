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

def CreateDataset(markermapFolder: Path, outputPath: Path, minLength: int=5, durationToExtract: int=15):
    allMarkermapFiles = [p for p in tqdm(markermapFolder.glob('**/*.markermap'), desc='searching *.markermap ...') if p.with_suffix('.ass.original').is_file()]
    logger.info(f'found {len(allMarkermapFiles)} files')

    cmTextList = []
    nonCmTextList = []
    for markermapPath in tqdm(allMarkermapFiles, desc='extracting cm/non-cm subtitles text ...'):
        with markermapPath.open() as f:
            markermap = json.load(f)
            assPath = markermapPath.with_suffix('.ass.original')
            assGenPath = markermapPath.with_suffix('.assgen')
            for k,v in markermap.items():
                clip = eval(k)
                # skip very short clips
                if clip[1] - clip[0] < minLength:
                    continue
                # only take first "durationToExtract" (15) seconds
                if clip[1] - clip[0] > durationToExtract:
                    clip15s = (clip[0], clip[0] + durationToExtract)
                    text = ExtractSubtitlesText(assPath, clip15s)
                else:
                    text = ExtractSubtitlesText(assPath, clip)
                # extract text from speech recognization
                if len(text) == 0:
                    text = ExtractGenSubtitlesText(assGenPath, clip)
                # skip spaces
                if text.isspace():
                    continue
                if v['_groundtruth'] == 0.0 and v['subtitles'] == 1.0:
                    cmTextList.append(text)
                elif v['_groundtruth'] == 1.0 and v['subtitles'] == 1.0:
                    nonCmTextList.append(text)

    cmTextList = list(set(cmTextList))
    nonCmTextList = list(set(nonCmTextList))

    logger.info(f'CM: {len(cmTextList)}, Non-CM: {len(nonCmTextList)}')
    with outputPath.open('w') as f:
        json.dump({ 'cm' : cmTextList, "noncm" : nonCmTextList }, f, indent=True, ensure_ascii=False)
