import pytest
from tests import samplesDir, conan_ts, conan_ptsmap, conan_markermap, conan_metadata, conan_dataset
import tsmarker.ensemble

def test_CreateDataset_Success():
    csvPath = samplesDir / 'conan.csv'
    tsmarker.ensemble.CreateDataset(
        folder=conan_metadata,
        csvPath=csvPath,
        properties=[ 'subtitles', 'position', 'duration', 'duration_prev', 'duration_next'])
    assert csvPath.is_file()
    csvPath.unlink()

def test_Train_Success():
    dataset = tsmarker.ensemble.LoadDataset(csvPath=conan_dataset)
    clf = tsmarker.ensemble.Train(dataset)
    assert clf is not None

def test_MarkByEnsemble_Success():
    dataset = tsmarker.ensemble.LoadDataset(csvPath=conan_dataset)
    clf = tsmarker.ensemble.Train(dataset)
    model = clf, dataset['columns']
    tsmarker.ensemble.Mark(model=model, markerPath=conan_markermap, dryrun=True)
