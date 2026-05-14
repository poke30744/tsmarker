import json, tempfile
from pathlib import Path
from tsmarker.ptsmap import PtsMap
from tsmarker.common import MarkerMap, _merge_neighbors, _auto_by_method, _clips_duration, _split_clips


def _make_temp_ptsmap(ptsmap_data):
    fd, path = tempfile.mkstemp(suffix='.ptsmap')
    with open(fd, 'w') as f:
        json.dump(ptsmap_data, f)
    return PtsMap(Path(path)), Path(path)


def test_merge_neighbors():
    clips = [(0.0, 100.0), (100.0, 200.0), (300.0, 400.0)]
    result = _merge_neighbors(clips)
    assert result == [(0.0, 200.0), (300.0, 400.0)]


def test_merge_neighbors_empty():
    assert _merge_neighbors([]) == []


def test_merge_neighbors_single():
    assert _merge_neighbors([(0.0, 100.0)]) == [(0.0, 100.0)]


def test_clips_duration():
    assert _clips_duration([(0.0, 100.0), (200.0, 300.0)]) == 200.0


def test_split_clips():
    clips = [(0.0, 50.0), (50.0, 100.0), (100.0, 150.0), (150.0, 200.0)]
    groups = _split_clips(clips, 2)
    assert len(groups) == 2
    assert sum(len(g) for g in groups) == 4
    all_clips = [c for g in groups for c in g]
    assert sorted(all_clips) == sorted(clips)


def test_MarkerMap_basic():
    ptsmap_data = {
        "0.0": {"prev_end_pts": 0.0, "next_start_pts": 0.0},
        "100.0": {"prev_end_pts": 100.0, "next_start_pts": 100.5},
        "200.0": {"prev_end_pts": 200.0, "next_start_pts": 200.0},
    }
    ptsMap, pts_path = _make_temp_ptsmap(ptsmap_data)
    try:
        # Use non-existent path so MarkerMap initializes from ptsMap.Clips()
        marker_path = Path(tempfile.mkdtemp()) / 'test.markermap'
        mm = MarkerMap(marker_path, ptsMap)
        clip = (0.0, 100.0)
        mm.Mark(clip, 'subtitles', 0.8)
        mm.Mark(clip, 'logo', 0.3)
        mm.Save()

        mm2 = MarkerMap(marker_path, ptsMap)
        assert mm2.Value(clip, 'subtitles') == 0.8
        assert mm2.Value(clip, 'logo') == 0.3
        assert 'subtitles' in mm2.Properties()
        assert 'logo' in mm2.Properties()

        marker_path.unlink(missing_ok=True)
        marker_path.parent.rmdir()
    finally:
        pts_path.unlink()


def test_MarkerMap_Normalized():
    ptsmap_data = {
        "0.0": {"prev_end_pts": 0.0, "next_start_pts": 0.0},
        "100.0": {"prev_end_pts": 100.0, "next_start_pts": 100.5},
        "200.0": {"prev_end_pts": 200.0, "next_start_pts": 200.0},
    }
    ptsMap, pts_path = _make_temp_ptsmap(ptsmap_data)
    try:
        marker_path = Path(tempfile.mkdtemp()) / 'test.markermap'
        mm = MarkerMap(marker_path, ptsMap)
        clips = mm.Clips()
        mm.Mark(clips[0], 'subtitles', 1.0)
        mm.Mark(clips[1], 'subtitles', 0.0)
        mm.Mark(clips[0], 'logo', 0.5)
        mm.Mark(clips[1], 'logo', 0.5)
        mm.Save()

        normalized = mm.Normalized()
        # subtitles: mean=0.5, std=0.5 → (1.0-0.5)/0.5=1.0, (0.0-0.5)/0.5=-1.0
        assert abs(normalized[str(clips[0])]['subtitles'] - 1.0) < 0.01
        assert abs(normalized[str(clips[1])]['subtitles'] - (-1.0)) < 0.01
        # logo: all equal → std=0 → mean subtracted, no division → 0
        assert normalized[str(clips[0])]['logo'] == 0.0

        marker_path.unlink(missing_ok=True)
        marker_path.parent.rmdir()
    finally:
        pts_path.unlink()


def _make_marker(ptsmap_data, marks):
    """Create a MarkerMap with marked clips, saved to a temp file."""
    ptsMap, pts_path = _make_temp_ptsmap(ptsmap_data)
    marker_path = Path(tempfile.mkdtemp()) / 'test.markermap'
    mm = MarkerMap(marker_path, ptsMap)
    for clip, prop, value in marks:
        mm.Mark(clip, prop, value)
    mm.Save()
    return MarkerMap(marker_path, ptsMap), pts_path, marker_path


def test_auto_by_method_subtitles():
    ptsmap_data = {
        "0.0": {"prev_end_pts": 0.0, "next_start_pts": 0.0},
        "100.0": {"prev_end_pts": 100.0, "next_start_pts": 100.5},
        "200.0": {"prev_end_pts": 200.0, "next_start_pts": 200.0},
    }
    mm, pts_path, marker_path = _make_marker(ptsmap_data, [
        ((0.0, 100.0), 'subtitles', 1.0),
        ((100.0, 200.0), 'subtitles', 0.2),
    ])
    try:
        assert _auto_by_method(mm) == 'subtitles'
    finally:
        pts_path.unlink()
        marker_path.unlink()
        marker_path.parent.rmdir()


def test_auto_by_method_logo_fallback():
    ptsmap_data = {
        "0.0": {"prev_end_pts": 0.0, "next_start_pts": 0.0},
        "100.0": {"prev_end_pts": 100.0, "next_start_pts": 100.5},
        "200.0": {"prev_end_pts": 200.0, "next_start_pts": 200.0},
    }
    mm, pts_path, marker_path = _make_marker(ptsmap_data, [
        ((0.0, 100.0), 'subtitles', 0.0),
        ((100.0, 200.0), 'subtitles', 0.5),
    ])
    try:
        assert _auto_by_method(mm) == 'logo'
    finally:
        pts_path.unlink()
        marker_path.unlink()
        marker_path.parent.rmdir()


def test_auto_by_method_groundtruth():
    ptsmap_data = {
        "0.0": {"prev_end_pts": 0.0, "next_start_pts": 0.0},
        "100.0": {"prev_end_pts": 100.0, "next_start_pts": 100.5},
        "200.0": {"prev_end_pts": 200.0, "next_start_pts": 200.0},
    }
    mm, pts_path, marker_path = _make_marker(ptsmap_data, [
        ((0.0, 100.0), '_groundtruth', 1.0),
        ((100.0, 200.0), '_groundtruth', 0.0),
        ((0.0, 100.0), 'subtitles', 0.2),
        ((100.0, 200.0), 'subtitles', 0.8),
    ])
    try:
        assert _auto_by_method(mm) == '_groundtruth'
    finally:
        pts_path.unlink()
        marker_path.unlink()
        marker_path.parent.rmdir()


def test_auto_by_method_speech():
    ptsmap_data = {
        "0.0": {"prev_end_pts": 0.0, "next_start_pts": 0.0},
        "100.0": {"prev_end_pts": 100.0, "next_start_pts": 100.5},
        "200.0": {"prev_end_pts": 200.0, "next_start_pts": 200.0},
    }
    mm, pts_path, marker_path = _make_marker(ptsmap_data, [
        ((0.0, 100.0), 'speech', 0.9),
        ((100.0, 200.0), 'speech', 0.1),
        ((0.0, 100.0), 'subtitles', 1.0),
        ((100.0, 200.0), 'subtitles', 0.0),
    ])
    try:
        assert _auto_by_method(mm) == 'speech'
    finally:
        pts_path.unlink()
        marker_path.unlink()
        marker_path.parent.rmdir()
