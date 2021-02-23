import pytest
from tests import samplesDir, conan_ts, conan_ptsmap, conan_markermap
import tsmarker.common

def test_SelectClips():
    ptsMap, _ = tsmarker.common.LoadExistingData(indexPath=conan_ptsmap, markerPath=conan_markermap)
    clips = tsmarker.common.GetClips(ptsMap)
    selectedClips, selectedLen = tsmarker.common.SelectClips(clips)
    assert len(selectedClips) > 0
    assert selectedLen > 0