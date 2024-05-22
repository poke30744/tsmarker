#!/usr/bin/env python3
import argparse, pickle, yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from . import common
from tscutter.common import PtsMap

logger = logging.getLogger('tsmarker.ensemble')

def LoadDescFeatures(path: Path) -> dict[str, float]:
    with path.open(encoding='utf-8') as f:
        desc = yaml.safe_load(f)
    serviceId = desc['serviceId']
    startAt = datetime.fromtimestamp(desc['startAt'] / 1000)
    start_hour = startAt.hour
    start_weekday = startAt.weekday()
    video_duration = desc['duration']
    genres_lv1 = desc['genres'][0]['lv1']
    genres_lv2 = desc['genres'][0]['lv2']
    genres_un1 = desc['genres'][0]['un1']
    genres_un2 = desc['genres'][0]['un2']
    folder_hash = int.from_bytes(hashlib.sha256(path.parent.stem.encode()).digest()[:4], 'little')
    name_hash = int.from_bytes(hashlib.sha256(path.stem.encode()).digest()[:4], 'little')
    return {
        'serviceId': serviceId,
        'start_hour': start_hour,
        'start_weekday': start_weekday,
        'video_duration': video_duration,
        'genres_lv1': genres_lv1,
        'genres_lv2': genres_lv2,
        'genres_un1': genres_un1,
        'genres_un2': genres_un2,
        'folder_hash': folder_hash,
        'name_hash': name_hash
    }

def LoadFeaturesFromVideo(path: Path, normalize: bool=False) -> list[dict[str, float]]:
    # load features from markermap
    markerMapPath = path.parent / '_metadata' / path.with_suffix('.markermap').name
    if not markerMapPath.exists():
        return []
    markerMap = MarkerMap(markerMapPath, None) # type: ignore
    # skip older data
    if len(markerMap.Properties()) < 9:
        return []
    # skip data without ground truth
    if not '_groundtruth' in markerMap.Properties():
        return []
    
    # load featuren from desc (same values for all clips)
    descPath = path.with_suffix('.yaml')
    descFeatures = LoadDescFeatures(descPath)
    
    result = []
    for clip, data in (markerMap.Normalized() if normalize else markerMap.data).items():
        del data['_ensemble']
        # append desc
        data.update(descFeatures)

        # append basic info
        data['_clip'], data['_filename'] = clip, path.stem
        result.append(data)

    return result
    
def CreateDataset(folder: Path, csvPath: Path, normalize: bool=False, quiet: bool=False):
    if csvPath.exists():
        # load existing csv
        df = pd.read_csv(csvPath)
        existingVideoFiles = df['_filename'].unique().tolist()
    else:
        df = pd.DataFrame()
        existingVideoFiles = []

    # find video files
    videoFiles = []
    videoFilesPathMap: dict[str, Path] = {}
    for path in tqdm(Path(folder).glob('**/*.mp4'), desc='searching *.mp4 ...', disable=quiet):
        videoFile = path.stem.replace('_trimmed', '').replace('_prog', '')
        videoFiles.append(videoFile)
        videoFilesPathMap[videoFile] = path
    logger.info(f'found {len(videoFiles)} video files.')

    # find updated files
    commonVideoFiles = set(existingVideoFiles) & set(videoFiles)
    newVideoFiles = set(videoFiles) - commonVideoFiles

    # keep only common files
    if len(commonVideoFiles) > 0:
        df = df[df['_filename'].isin(commonVideoFiles)]
    
    # load features from new files
    for videoFile in tqdm(newVideoFiles, desc='loading features from updated files ...', disable=quiet):
        path = videoFilesPathMap[videoFile]
        features = LoadFeaturesFromVideo(path, normalize)
        if len(features) == 0:
            df = pd.concat([df, pd.DataFrame({"_filename": [videoFile]})], ignore_index=True)
        else:
            df = pd.concat([df, pd.DataFrame(features)], ignore_index=True)

    if df is not None and csvPath is not None:
        df.to_csv(csvPath, encoding='utf-8-sig', index=False)
        logger.info(f'Updated dataset: {csvPath.absolute()}')
    return df

def LoadDataset(csvPath, columnsToExclude=[]):
    df = pd.read_csv(csvPath).dropna()
    columns = list(df.columns)
    columnsToExclude += [
        # always exclude below
        '_groundtruth',
        '_clip',
        '_filename',
        '_ensemble'
    ]
    for column in columnsToExclude:
        if column in columns:
            columns.remove(column)
    data = df[columns].to_numpy()
    target = df['_groundtruth'].to_numpy()
    return { 'data': data, 'target': target, 'columns': columns }

def Train(dataset, random_state=0, test_size=0.3, quiet=False):
    X, y = dataset['data'], dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state + 42)
    weight_train = compute_sample_weight(class_weight='balanced', y=y_train)
    weight_test = compute_sample_weight(class_weight='balanced', y=y_test)

    best_n, best_score = 0, 0
    for n in tqdm(range(1, 100), disable=quiet):
        clf = AdaBoostClassifier(n_estimators=n, random_state=random_state)
        clf.fit(X_train, y_train, sample_weight=np.copy(weight_train))
        score = clf.score(X_test, y_test, sample_weight=weight_test) # type: ignore
        if best_score < score:
            best_score = score
            best_n = n

    logger.info('best n: {}, best score: {}'.format(best_n, best_score))
    
    clf = AdaBoostClassifier(n_estimators=best_n, random_state=0)
    clf.fit(X_train, y_train, sample_weight=np.copy(weight_train))    
    return clf

class MarkerMap(common.MarkerMap):
    def MarkAll(self, model: tuple, normalize: bool=False, dryrun=False) -> None:
        # load file desc properties
        videoFolder = self.path.parent.parent
        videoPath = videoFolder / self.path.with_suffix('.mp4').name
        descPath = videoFolder / self.path.with_suffix('.yaml').name
        descFeatures = LoadDescFeatures(descPath)

        clf, columns = model
        for clip, data in (self.Normalized() if normalize else self.data).items():
            # append desc
            data.update(descFeatures)
            x = np.array([[ data[col] for col in columns ]])
            self.Mark(clip, '_ensemble', clf.predict(x)[0])
            if dryrun:
                logger.info(clip, clf.predict(x))
        if not dryrun:
            self.Save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict clips using ensemble models')    
    subparsers = parser.add_subparsers(required=True, title='subcommands', dest='command')

    subparser = subparsers.add_parser('dataset', help='create CSV dataset from .markermap files')
    subparser.add_argument('--input', '-i', required=True, help='the folder where to search *.markermap files')
    subparser.add_argument('--output', '-o', required=True, help='output csv path')

    subparser = subparsers.add_parser('train', help='train the model')
    subparser.add_argument('--input', '-i', required=True, help='dataset CSV path')
    subparser.add_argument('--output', '-o', required=True, help='the model path to output')

    subparser = subparsers.add_parser('predict', help='predict using the trained mode')
    subparser.add_argument('--model', '-m', required=True, help='the model path')
    subparser.add_argument('--input', '-i', required=True, help='the path of .markermap file')
    subparser.add_argument('--dryrun', '-d', action='store_true', help='do not modify the .markermap file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.command == 'dataset':
        CreateDataset(folder=Path(args.input), csvPath=Path(args.output))
        # print summary
        dataset = LoadDataset(csvPath=Path(args.output))
        dataShape, targetShape = dataset['data'].shape, dataset['target'].shape
        print(f'data shape: {dataShape}, target shape: {targetShape}')
        targetElements = np.unique(dataset['target'], return_counts=True)
        print(f'columns: {dataset["columns"]}')
        print(f'target counts: {targetElements[0][0]}: {targetElements[1][0]}, {targetElements[0][1]}: {targetElements[1][1]}')
        sampleWeight = compute_sample_weight(class_weight='balanced', y=dataset['target'])
        weightElements = np.unique(sampleWeight, return_counts=True)
        print(f'sample weights: {weightElements[0][0]}: {weightElements[1][0]}, {weightElements[0][1]}: {weightElements[1][1]}')
    elif args.command == 'train':
        dataset = LoadDataset(csvPath=Path(args.input))
        columns = dataset['columns']
        clf = Train(dataset)
        modelPath = Path(args.output)
        with modelPath.open('wb') as f:
            pickle.dump((clf, columns), f)
            logger.info(f'Saved model: {modelPath.absolute()}')
    elif args.command == 'predict':
        with open(args.model, 'rb') as f:
            model = pickle.load(f)
        MarkerMap(Path(args.input), PtsMap(Path(args.input).with_suffix('.ptsmap'))).MarkAll(model, dryrun=args.dryrun)
