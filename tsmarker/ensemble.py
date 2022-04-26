#!/usr/bin/env python3
import argparse, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from .common import SaveMarkerMap

logger = logging.getLogger('tsmarker.ensemble')

def CreateDataset(folder, csvPath, properties):
    df = None
    properties.append('_groundtruth')
    properties.append('_ensemble')
    for path in tqdm(sorted(list(Path(folder).glob('**/*.markermap')))):
        with path.open() as f:
            markermap = json.load(f)
            for clip, data in markermap.items():
                for k in list(data.keys()):
                    if not k in properties:
                        del data[k]
                data['_clip'], data['_filename'] = clip, path.name
                df = pd.DataFrame(data, index=[0]) if df is None else df.append(pd.DataFrame(data, index=[len(df)]))
    if df is not None and csvPath is not None:
        df.to_csv(csvPath, encoding='utf-8-sig')
    return df

def LoadDataset(csvPath, columnsToExclude=[]):
    df = pd.read_csv(csvPath).fillna(0)
    columns = list(df.columns)
    columnsToExclude += [
        # always exclude below
        'Unnamed: 0',
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

def Train(dataset, random_state=0, test_size=0.3):
    X, y = dataset['data'], dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state + 42)
    weight_train = compute_sample_weight(class_weight='balanced', y=y_train)
    weight_test = compute_sample_weight(class_weight='balanced', y=y_test)

    best_n, best_score = 0, 0
    for n in tqdm(range(1, 100)):
        clf = AdaBoostClassifier(n_estimators=n, random_state=random_state)
        clf.fit(X_train, y_train, sample_weight=np.copy(weight_train))
        score = clf.score(X_test, y_test, sample_weight=weight_test)
        if best_score < score:
            best_score = score
            best_n = n

    logger.info('best n: {}, best score: {}'.format(best_n, best_score))
    
    clf = AdaBoostClassifier(n_estimators=best_n, random_state=0)
    clf.fit(X_train, y_train, sample_weight=np.copy(weight_train))    
    return clf

def Mark(model, markerPath, dryrun=False):
    markerPath = Path(markerPath)
    clf, columns = model
    with markerPath.open() as f:
        markerMap = json.load(f)
    for k, v in markerMap.items():
        x = np.array([[ v[col] for col in columns ]])
        v['_ensemble'] = clf.predict(x)[0]
        if dryrun:
            logger.info(k, clf.predict(x))
    if not dryrun:
        markerPath = SaveMarkerMap(markerMap, markerPath)
        return markerPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict clips using ensemble models')    
    subparsers = parser.add_subparsers(required=True, title='subcommands', dest='command')

    subparser = subparsers.add_parser('dataset', help='create CSV dataset from .markermap files')
    subparser.add_argument('--input', '-i', required=True, help='the folder where to search *.markermap files')
    subparser.add_argument('--output', '-o', required=True, help='output csv path')
    subparser.add_argument('--properties', '-p', nargs='+', help='properties in .markermap files')

    subparser = subparsers.add_parser('train', help='train the model')
    subparser.add_argument('--input', '-i', required=True, help='dataset CSV path')
    subparser.add_argument('--output', '-o', required=True, help='the model path to output')

    subparser = subparsers.add_parser('predict', help='predict using the trained mode')
    subparser.add_argument('--model', '-m', required=True, help='the model path')
    subparser.add_argument('--input', '-i', required=True, help='the path of .markermap file')

    args = parser.parse_args()

    if args.command == 'dataset':
        CreateDataset(folder=args.input, csvPath=args.output, properties=args.properties)
        # print summary
        dataset = LoadDataset(csvPath=args.output)
        dataShape, targetShape = dataset['data'].shape, dataset['target'].shape
        print(f'data shape: {dataShape}, target shape: {targetShape}')
        targetElements = np.unique(dataset['target'], return_counts=True)
        print(f'columns: {dataset["columns"]}')
        print(f'target counts: {targetElements[0][0]}: {targetElements[1][0]}, {targetElements[0][1]}: {targetElements[1][1]}')
        sampleWeight = compute_sample_weight(class_weight='balanced', y=dataset['target'])
        weightElements = np.unique(sampleWeight, return_counts=True)
        print(f'sample weights: {weightElements[0][0]}: {weightElements[1][0]}, {weightElements[0][1]}: {weightElements[1][1]}')
    elif args.command == 'train':
        dataset = LoadDataset(csvPath=args.input)
        columns = dataset['columns']
        clf = Train(dataset)
        with open(args.output, 'wb') as f:
            pickle.dump((clf, columns), f)
    elif args.command == 'predict':
        with open(args.model, 'rb') as f:
            model = pickle.load(f)
        Mark(model, args.input, dryrun=True)
