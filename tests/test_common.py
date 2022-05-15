import pytest
from . import samplesDir, conan_ts, conan_ptsmap, conan_markermap, edges_with_boarder1, edges_with_boarder2
from tscutter.common import PtsMap
from tsmarker import logo
from tsmarker import common
import cv2 as cv 

def test_SelectClips():
    selectedClips, selectedLen = PtsMap(conan_ptsmap).SelectClips()
    assert len(selectedClips) > 0
    assert selectedLen > 0

def test_RemoveBoarder():
    edges = cv.imread(str(edges_with_boarder1), 0)
    logo.RemoveBoarder(edges)
    cv.imwrite(str(edges_with_boarder1.with_suffix('.noboarder.png')), edges)

    edges = cv.imread(str(edges_with_boarder2), 0)
    logo.RemoveBoarder(edges)
    cv.imwrite(str(edges_with_boarder2.with_suffix('.noboarder.png')), edges)

def test_MarkerMap_Properties():
    markerMap = common.MarkerMap(conan_markermap, None)
    assert len(markerMap.Properties()) > 0
