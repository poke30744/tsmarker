import pytest
from . import salor_moon_C_02_ts, salor_moon_C_02_ptsmap, invalid_ts, not_existing_ts
from . import samplesDir, conan_ts, conan_ptsmap, conan_markermap
from tscutter.common import InvalidTsFormat, PtsMap
from tsmarker.marker import MarkVideo
from tsmarker.common import MarkerMap
from tsmarker import groundtruth
import shutil, json

def test_MarkBySubtitles_Success():
    markerPath = salor_moon_C_02_ts.with_suffix('.markermap')
    MarkVideo(
        videoPath=salor_moon_C_02_ts, 
        indexPath=salor_moon_C_02_ptsmap, 
        markerPath=markerPath, 
        methods=['subtitles'])
    assert markerPath.is_file()
    assert markerPath.stat().st_size > 0
    markerPath.unlink()

def test_MarkBySubtitles_Invalid():
    with pytest.raises(json.decoder.JSONDecodeError):
        MarkVideo(
            videoPath=invalid_ts, 
            indexPath=invalid_ts, 
            markerPath=None, 
            methods=['subtitles'])
    
def test_MarkByClipinfo_Success():
    markerPath = salor_moon_C_02_ts.with_suffix('.markermap')
    MarkVideo(
        videoPath=salor_moon_C_02_ts, 
        indexPath=salor_moon_C_02_ptsmap, 
        markerPath=markerPath, 
        methods=['clipinfo'])
    assert markerPath.is_file()
    assert markerPath.stat().st_size > 0
    markerPath.unlink()

def test_MarkByClipinfo_Invalid():
    with pytest.raises(InvalidTsFormat, match='"invalid.ts" is invalid!'):
        MarkVideo(
            videoPath=invalid_ts, 
            indexPath=salor_moon_C_02_ptsmap, 
            markerPath=None, 
            methods=['clipinfo'])

def test_CutCMs_Success():
    markerMap = MarkerMap(conan_markermap, PtsMap(conan_ptsmap))
    outputFolder = samplesDir / 'conan'
    markerMap.Cut(videoPath=conan_ts, byMethod='subtitles', outputFolder=outputFolder)
    assert outputFolder.is_dir()
    shutil.rmtree(outputFolder)

def test_MarkGroundTruth_Success():
    outputFolder = samplesDir / 'conan'
    MarkerMap(conan_markermap, PtsMap(conan_ptsmap)).Cut(videoPath=conan_ts, byMethod='subtitles', outputFolder=outputFolder)
    assert outputFolder.is_dir()
    groundtruth.MarkerMap(conan_markermap,  PtsMap(conan_ptsmap)).MarkAll(clipsFolder=outputFolder)
    shutil.rmtree(outputFolder)