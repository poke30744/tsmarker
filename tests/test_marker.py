import json, tempfile
from pathlib import Path
from unittest.mock import patch
from tscutter.common import PtsMap
from tsmarker.common import MarkerMap, get_program_clips


def _make_temp_ptsmap(ptsmap_data):
    fd, path = tempfile.mkstemp(suffix='.ptsmap')
    with open(fd, 'w') as f:
        json.dump(ptsmap_data, f)
    return PtsMap(Path(path)), Path(path)


def test_MarkerMap_clips():
    ptsmap_data = {
        "0.0": {"prev_end_pts": 0.0, "next_start_pts": 0.0},
        "100.0": {"prev_end_pts": 99.0, "next_start_pts": 101.0},
        "200.0": {"prev_end_pts": 200.0, "next_start_pts": 200.0},
    }
    ptsMap, pts_path = _make_temp_ptsmap(ptsmap_data)
    try:
        marker_path = Path(tempfile.mkdtemp()) / 'test.markermap'
        mm = MarkerMap(marker_path, ptsMap)
        clips = mm.Clips()
        assert len(clips) == 2
        assert clips[0] == (0.0, 100.0)
        assert clips[1] == (100.0, 200.0)
        marker_path.unlink(missing_ok=True)
        marker_path.parent.rmdir()
    finally:
        pts_path.unlink()


def test_MarkerMap_cut_logic():
    ptsmap_data = {
        "0.0": {"prev_end_pts": 0.0, "next_start_pts": 0.0, "prev_end_sad": 0.0, "next_start_sad": 0.0},
        "100.0": {"prev_end_pts": 99.0, "next_start_pts": 101.0, "prev_end_sad": 0.1, "next_start_sad": 0.2},
        "200.0": {"prev_end_pts": 200.0, "next_start_pts": 200.0, "prev_end_sad": 0.0, "next_start_sad": 0.0},
    }
    ptsMap, pts_path = _make_temp_ptsmap(ptsmap_data)
    try:
        marker_path = Path(tempfile.mkdtemp()) / 'test.markermap'
        mm = MarkerMap(marker_path, ptsMap)
        clip0 = (0.0, 100.0)
        clip1 = (100.0, 200.0)
        mm.Mark(clip0, 'subtitles', 1.0)
        mm.Mark(clip0, 'logo', 0.8)
        mm.Mark(clip1, 'subtitles', 0.0)
        mm.Mark(clip1, 'logo', 0.1)
        mm.Save()

        with patch('subprocess.run') as mock_run:
            # Create expected output files so shutil.move works
            def create_files(*args, **kwargs):
                cmd = args[0]
                out_file = Path(cmd[-1])  # last arg is output file
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.touch()
            mock_run.side_effect = create_files

            with tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir) / 'test_cut'
                mm.Cut(Path('/fake/video.ts'), 'subtitles', out)

                assert mock_run.call_count == 2
                call_args = mock_run.call_args_list[0][0][0]
                ss_idx = call_args.index('-ss')
                assert call_args[ss_idx + 1] == '0.0'
                to_idx = call_args.index('-to')
                assert call_args[to_idx + 1] == '99.0'

                cm_files = list((out / 'CM').glob('*.ts'))
                program_files = [f for f in out.glob('*.ts') if f.parent.name != 'CM']
                assert len(cm_files) == 1
                assert len(program_files) == 1

        marker_path.unlink(missing_ok=True)
        marker_path.parent.rmdir()
    finally:
        pts_path.unlink()


def test_get_program_clips_by_group():
    ptsmap_data = {
        "0.0": {"prev_end_pts": 0.0, "next_start_pts": 0.0},
        "50.0": {"prev_end_pts": 49.0, "next_start_pts": 51.0},
        "100.0": {"prev_end_pts": 99.0, "next_start_pts": 101.0},
        "200.0": {"prev_end_pts": 200.0, "next_start_pts": 200.0},
    }
    ptsMap, pts_path = _make_temp_ptsmap(ptsmap_data)
    try:
        marker_path = Path(tempfile.mkdtemp()) / 'test.markermap'
        mm = MarkerMap(marker_path, ptsMap)
        clips = mm.Clips()
        mm.Mark(clips[0], 'subtitles', 1.0)
        mm.Mark(clips[1], 'subtitles', 1.0)
        mm.Mark(clips[2], 'subtitles', 0.0)
        mm.Save()

        result = get_program_clips(marker_path, pts_path, by='subtitles', split=1, by_group=False)
        assert result['by_method'] == 'subtitles'
        # Two program clips are neighbors → merged
        assert result['groups'] == [[(0.0, 100.0)]]

        result2 = get_program_clips(marker_path, pts_path, by='subtitles', split=1, by_group=True)
        # by_group: each merged clip is its own group → 1 merged clip = 1 group
        assert result2['groups'] == [[(0.0, 100.0)]]

        marker_path.unlink(missing_ok=True)
        marker_path.parent.rmdir()
    finally:
        pts_path.unlink()
