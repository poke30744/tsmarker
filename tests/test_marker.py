import pytest
from tests import salor_moon_C_02_ts, salor_moon_C_02_ptsmap, invalid_ts, not_existing_ts
from tests import samplesDir, conan_ts, conan_ptsmap, conan_markermap
import tsmarker.marker
import tsutils
import shutil, json

def test_MarkBySubtitles_Success():
    markerPath = salor_moon_C_02_ts.with_suffix('.markermap')
    tsmarker.marker.MarkVideo(
        videoPath=salor_moon_C_02_ts, 
        indexPath=salor_moon_C_02_ptsmap, 
        markerPath=markerPath, 
        methods=['subtitles'])
    assert markerPath.is_file()
    assert markerPath.stat().st_size > 0
    markerPath.unlink()

def test_MarkBySubtitles_Invalid():
    with pytest.raises(json.decoder.JSONDecodeError):
        tsmarker.marker.MarkVideo(
            videoPath=invalid_ts, 
            indexPath=invalid_ts, 
            markerPath=None, 
            methods=['subtitles'])
    
def test_MarkByClipinfo_Success():
    markerPath = salor_moon_C_02_ts.with_suffix('.markermap')
    tsmarker.marker.MarkVideo(
        videoPath=salor_moon_C_02_ts, 
        indexPath=salor_moon_C_02_ptsmap, 
        markerPath=markerPath, 
        methods=['clipinfo'])
    assert markerPath.is_file()
    assert markerPath.stat().st_size > 0
    markerPath.unlink()

def test_MarkByClipinfo_Invalid():
    with pytest.raises(tsutils.InvalidTsFormat, match='"invalid.ts" is invalid!'):
        tsmarker.marker.MarkVideo(
            videoPath=invalid_ts, 
            indexPath=salor_moon_C_02_ptsmap, 
            markerPath=None, 
            methods=['clipinfo'])

def test_CutCMs_Success():
    outputFolder = tsmarker.marker.CutCMs(
        videoPath=conan_ts, 
        indexPath=conan_ptsmap, 
        markerPath=conan_markermap, 
        byMethod='subtitles', 
        outputFolder=samplesDir / 'conan')
    assert outputFolder.is_dir()
    shutil.rmtree(outputFolder)

def test_MarkGroundTruth_Success():
    outputFolder = tsmarker.marker.CutCMs(
        videoPath=conan_ts, 
        indexPath=conan_ptsmap, 
        markerPath=conan_markermap, 
        byMethod='subtitles', 
        outputFolder=samplesDir / 'conan')
    tsmarker.marker.MarkGroundTruth(outputFolder, conan_markermap)
    assert outputFolder.is_dir()
    shutil.rmtree(outputFolder)