import logging
from pathlib import Path
import shutil
import tempfile, requests, json
from tqdm import tqdm
import speech_recognition as sr
from .. import common
from .dataset import ExtractSubtitlesText
from ..subtitles import Extract
from tscutter.ffmpeg import InputFile
from tscutter.common import PtsMap

logger = logging.getLogger('tsmarker.speech')

def ExtractAudioText(videoPath: Path, clip: tuple[float, float]) -> str:
    recognizer = sr.Recognizer()
    inputFile = InputFile(videoPath)
    with tempfile.TemporaryDirectory(prefix='ExtractAudioText_') as tmpFolder:
        inputFile.ExtractStream(output=Path(tmpFolder), ss=clip[0], to=clip[1], toWav=True, videoTracks=[], audioTracks=[0], quiet=True)
        audioFilename = Path(tmpFolder) / 'audio_0.wav'
        try:
            with sr.AudioFile(str(audioFilename)) as source:
                audio = recognizer.record(source)
        except ValueError: 
            return ''
    try:
        text = recognizer.recognize_google(audio, language='ja-JP')
        return text
    except sr.UnknownValueError:
        return ''
    except sr.RequestError:
        return ''

def PrepareSubtitles(videoPath: Path, ptsMap: PtsMap, quiet: bool=False):
    originalSubtitlesPath =  ptsMap.path.with_suffix('.ass.original')
    with tempfile.TemporaryDirectory(prefix='ExtractSubtitles_') as tmpFolder:
        for sub in Extract(videoPath, Path(tmpFolder)):
            if sub.suffix == '.ass':
                shutil.copy(sub, originalSubtitlesPath)
                break
    clips = ptsMap.Clips()
    textList = [ ExtractSubtitlesText(originalSubtitlesPath, clip) for clip in clips ] if originalSubtitlesPath.exists() else [ '' ] * len(clips)
    # for clips without subtitles, extrac it from audio
    generatedSubtitlesPath = ptsMap.path.with_suffix('.assgen')
    generatedSubtitles = {}
    for i in tqdm(range(len(clips)), disable=quiet):
        if textList[i] == '':
            textList[i] = ExtractAudioText(videoPath, clips[i])
            generatedSubtitles[str(clips[i])] = textList[i]
    # save generated subtitles
    with generatedSubtitlesPath.open('w') as f:
        json.dump(generatedSubtitles, f, ensure_ascii=False, indent=True)

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, apiUrl: str, quiet=False) -> None:
        originalSubtitlesPath = self.path.with_suffix('.ass.original')
        generatedSubtitlesPath = self.path.with_suffix('.assgen')
        if not originalSubtitlesPath.exists() or not generatedSubtitlesPath.exists():
            PrepareSubtitles(videoPath, self.ptsMap, quiet=quiet)

        clips = self.Clips()    
        textList = [ ExtractSubtitlesText(originalSubtitlesPath, clip) for clip in clips ] if originalSubtitlesPath.exists() else [ '' ] * len(clips)
        with generatedSubtitlesPath.open() as f:
            generatedSubtitles = json.load(f)
        for i in range(len(clips)):
            if textList[i] == '':
                textList[i] = generatedSubtitles[str(clips[i])]

        # get prediction from RESTful service
        payload = json.dumps(textList)
        response = requests.request("POST", apiUrl, data=payload, headers={ 'Content-Type': 'application/json' })
        Y = response.json()
        for i in range(len(clips)):
            self.Mark(clips[i], 'speech', float(Y[i][0]))
        self.Save()

def ReMarkAll(markermapFolder: Path, apiUrl: str, quiet=False):
    files = []
    for markerPath in tqdm(markermapFolder.glob('**/*.markermap'), desc='searching *.markermap ...', disable=quiet):
        indexPath = markerPath.with_suffix('.ptsmap')
        originalSubtitlesPath = markerPath.with_suffix('.ass.original')
        generatedSubtitlesPath = markerPath.with_suffix('.assgen')
        if indexPath.exists() and originalSubtitlesPath.exists() and generatedSubtitlesPath.exists():
            files.append(markerPath)
    logger.info(f"Will re-mark {len(files)} files")
    for markerPath in tqdm(files, desc='re-marking ...', disable=quiet):
        indexPath = markerPath.with_suffix('.ptsmap')
        markermap = MarkerMap(markerPath, PtsMap(indexPath))
        markermap.MarkAll(Path(), apiUrl, quiet=quiet)