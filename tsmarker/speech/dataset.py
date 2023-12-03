from functools import lru_cache
import logging, re, json
from pathlib import Path
from tqdm import tqdm
import pysubs2

logger = logging.getLogger('tsmarker.dataset')

@lru_cache(maxsize=2)
def ExtractSubs(assPath: Path):
    return pysubs2.load(assPath)

def ExtractSubtitlesText(assPath: Path, clip: tuple[float, float]) -> str:
    result = []
    for event in ExtractSubs(assPath):
        start, end = clip[0] * 1000, clip[1] * 1000
        # if overlapped
        if event.start < end and start < event.end:
            text = event.text
            text = re.sub(r'\{.*?\}', '', text)
            text = text.replace(r'\N', '')
            result.append(text)
    return ' '.join(result)

def CreateDataset(markermapFolder: Path, outputPath: Path, minLength: int=5, durationToExtract: int=15):
    allMarkermapFiles = [p for p in tqdm(markermapFolder.glob('**/*.markermap'), desc='searching *.markermap ...') if p.with_suffix('.ass.original').is_file()]
    logger.info(f'found {len(allMarkermapFiles)} files')

    fileList = []
    for markermapPath in tqdm(allMarkermapFiles, desc='checking subtitles ...'):
        with markermapPath.open() as f:
            markermap = json.load(f)
            for k,v in markermap.items():
                if '_groundtruth' in v and 'subtitles' in v and  v['_groundtruth'] == 0.0 and v['subtitles'] == 1.0:
                    fileList.append(markermapPath)
                    break
    logger.info(f'found {len(fileList)} files with CM subtitles')

    cmTextList = []
    nonCmTextList = []
    for markermapPath in tqdm(fileList, desc='extracting cm/non-cm subtitles text ...'):
        with markermapPath.open() as f:
            markermap = json.load(f)
            assPath = markermapPath.with_suffix('.ass.original')
            for k,v in markermap.items():
                clip = eval(k)
                # skip very short clips
                if clip[1] - clip[0] < minLength:
                    continue
                # only take first "durationToExtract" (15) seconds
                if clip[1] - clip[0] > durationToExtract:
                    clip = (clip[0], clip[0] + durationToExtract)
                text = ExtractSubtitlesText(assPath, clip)
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
