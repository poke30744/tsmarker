# tsmarker

Mark commercial/program segments in MPEG2-TS video files using multiple detection methods.

## CLI Commands

```
tsmarker [--quiet] [--progress] [--version] COMMAND [ARGS]...
```

| Command | Description | Input | Output |
|---|---|---|---|
| `mark` | Mark clips by method(s) | TS + `.ptsmap` | `.markermap` (in-place) |
| `cut` | Split TS by marking result | TS + `.ptsmap` + `.markermap` | CM/ classified files |
| `groundtruth` | Manual verification mark | `.markermap` + clips folder | `.markermap` (in-place) + stdout JSON |
| `extract-clips` | Extract TS byte ranges | TS + `.ptsmap` + clips JSON | stdout TS or `.ts` files |
| `get-program-clips` | Group program clips | `.markermap` + `.ptsmap` | stdout JSON |
| `extract-logo` | Logo edge detection | TS + `.ptsmap` | PNG |
| `crop-detect` | Detect crop parameters | logo PNG | stdout JSON |
| `prepare-subtitles` | Extract subtitles + STT | TS + `.ptsmap` | `.ass.original` + `.assgen` |
| `ensemble-dataset` | Generate training CSV | search folder | CSV |
| `ensemble-train` | Train ensemble model | CSV | model.pkl |
| `ensemble-predict` | Predict with model | model + `.ptsmap` + `.markermap` | `.markermap` (in-place) |

### Mark methods

`--method`: `subtitles`, `clipinfo`, `logo`, `speech`

```bash
tsmarker mark --method subtitles --method clipinfo --method logo --method speech -i video.ts -x index.ptsmap -m output.markermap
```

## Dependencies

- Python ≥3.13
- tscutter (CLI, used for list-clips/select-clips/probe)
- ffmpeg, ffprobe (subtitle extraction via libaribcaption)
- scikit-learn, opencv-python, pandas, pysubs2, SpeechRecognition, openai, requests
