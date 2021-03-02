import pytest
from tests import samplesDir, conan_ts, conan_ptsmap, conan_markermap, edges_with_boarder1, edges_with_boarder2
import tsmarker.common
import cv2 as cv 

def test_SelectClips():
    ptsMap, _ = tsmarker.common.LoadExistingData(indexPath=conan_ptsmap, markerPath=conan_markermap)
    clips = tsmarker.common.GetClips(ptsMap)
    selectedClips, selectedLen = tsmarker.common.SelectClips(clips)
    assert len(selectedClips) > 0
    assert selectedLen > 0

def test_RemoveBoarder():
    edges = cv.imread(str(edges_with_boarder1), 0)
    tsmarker.common.RemoveBoarder(edges)
    cv.imwrite(str(edges_with_boarder1.with_suffix('.noboarder.png')), edges)

    edges = cv.imread(str(edges_with_boarder2), 0)
    tsmarker.common.RemoveBoarder(edges)
    cv.imwrite(str(edges_with_boarder2.with_suffix('.noboarder.png')), edges)