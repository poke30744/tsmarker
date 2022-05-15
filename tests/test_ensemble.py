import pytest
from . import samplesDir, conan_ts, conan_ptsmap, conan_markermap, conan_metadata, conan_dataset
from tscutter.common import PtsMap
from tsmarker import ensemble

def test_CreateDataset_Success():
    csvPath = samplesDir / 'conan.csv'
    ensemble.CreateDataset(
        folder=conan_metadata,
        csvPath=csvPath,
        properties=[ 'subtitles', 'position', 'duration', 'duration_prev', 'duration_next'])
    assert csvPath.is_file()
    csvPath.unlink()

def test_Train_Success():
    dataset = ensemble.LoadDataset(csvPath=conan_dataset)
    clf = ensemble.Train(dataset)
    assert clf is not None

def test_MarkByEnsemble_Success():
    dataset = ensemble.LoadDataset(csvPath=conan_dataset)
    clf = ensemble.Train(dataset)
    model = clf, dataset['columns']
    ensemble.MarkerMap(conan_markermap, PtsMap(conan_ptsmap)).MarkAll(model=model, dryrun=True)
