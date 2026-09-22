import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import speech_recognition as sr
from tscutter.common import PtsMap

from tsmarker.speech import text_extractor


class FakeRecognizer:
    """Recognizer that fails ``failures`` times before returning a transcription."""

    def __init__(self, failures: int = 0, text: str = 'テスト'):
        self.failures = failures
        self.text = text
        self.calls = 0

    def record(self, source):
        return b'audio'

    def recognize_google(self, audio, language=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise sr.RequestError('recognition request failed: Bad Request')
        return self.text


def _extract_audio_text(recognizer: FakeRecognizer, tmpPath: Path):
    """Run ExtractAudioText with ffmpeg and the real recognizer replaced."""
    with patch.object(text_extractor, 'InputFile') as inputFile, \
         patch.object(text_extractor, 'subprocess'), \
         patch.object(text_extractor.sr, 'AudioFile', MagicMock()), \
         patch.object(text_extractor.sr, 'Recognizer', return_value=recognizer), \
         patch.object(text_extractor.time, 'sleep') as sleep:
        inputFile.return_value.ffmpeg = 'ffmpeg'
        text = text_extractor.ExtractAudioText(tmpPath / 'video.m2ts', (0.0, 60.0))
    return text, sleep


def test_extract_audio_text_retries_request_error(tmp_path):
    recognizer = FakeRecognizer(failures=2)
    text, sleep = _extract_audio_text(recognizer, tmp_path)
    assert text == 'テスト'
    assert recognizer.calls == 3
    assert [call.args[0] for call in sleep.call_args_list] == text_extractor.RETRY_DELAYS[:2]


def test_extract_audio_text_gives_up_after_last_retry(tmp_path):
    recognizer = FakeRecognizer(failures=99)
    with pytest.raises(RuntimeError, match='Speech recognition failed'):
        _extract_audio_text(recognizer, tmp_path)
    assert recognizer.calls == len(text_extractor.RETRY_DELAYS) + 1


def _ptsmap(tmpPath: Path) -> PtsMap:
    index = tmpPath / 'video.ptsmap'
    index.write_text(json.dumps({'0.0': {}, '60.0': {}, '120.0': {}, '180.0': {}}), encoding='utf-8')
    return PtsMap(index)


def test_prepare_subtitles_keeps_transcribed_clips(tmp_path):
    ptsMap = _ptsmap(tmp_path)
    videoPath = tmp_path / 'video.m2ts'
    with patch.object(text_extractor, 'Extract', return_value=[]), \
         patch.object(text_extractor, 'ExtractAudioText') as extract:
        extract.side_effect = ['clip0', 'clip1', RuntimeError('boom')]
        with pytest.raises(RuntimeError):
            text_extractor.PrepareSubtitles(videoPath, ptsMap)

        # the two clips transcribed before the failure are kept
        generated = json.loads((tmp_path / 'video.assgen').read_text(encoding='utf-8'))
        assert generated == {'(0.0, 60.0)': 'clip0', '(60.0, 120.0)': 'clip1'}

        # the next run only transcribes the missing clip
        extract.reset_mock()
        extract.side_effect = None
        extract.return_value = 'clip2'
        text_extractor.PrepareSubtitles(videoPath, ptsMap)

    assert extract.call_count == 1
    assert extract.call_args.args[1] == (120.0, 180.0)
    assert json.loads((tmp_path / 'video.assgen').read_text(encoding='utf-8')) == {
        '(0.0, 60.0)': 'clip0', '(60.0, 120.0)': 'clip1', '(120.0, 180.0)': 'clip2'}
