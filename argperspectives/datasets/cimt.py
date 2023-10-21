from pathlib import Path
from typing import Container, Optional

import json

import pandas as pd
import datasets
import scipy
import transformers

class Dataset(object):
    """Loads disaggregated version of CIMT PartEval Argument Concreteness Corpus."""

    def __init__(self, features, annotators_mapping, sociodemographic_mapping) -> None:
        self.annotators_mapping = annotators_mapping
        self.sociodemographic_mapping = sociodemographic_mapping
        self.features = features

    @classmethod
    def _pad_labels_annotators(cls, df):
        annotations_max_length = df['annotator_indecies'].apply(len).max()
        df['annotator_indecies'] = df['annotator_indecies'].apply(lambda x: x + [-1] * (annotations_max_length - len(x)))
        df['labels'] = df['labels'].apply(lambda x: x + [-1] * (annotations_max_length - len(x)))
        return df

    @classmethod
    def load(
            cls,
            path_train,
            path_dev = None,
            path_test = None,
            do_majority_aggregation = False,
            preprocess = None,
            model_name : str = 'distilbert-base-uncased',
            n: Optional[int] = None,
            only_allow_annotators: Container[int] = None
        ):

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        max_length = tokenizer.model_max_length

        train = cls._load_preprocessed(path_train, 
                                       n = n, 
                                       preprocess = preprocess,
                                       only_allow_annotators=only_allow_annotators)
        split_dfs = [('train', train)]
        
        if path_dev:
            dev = cls._load_preprocessed(path_dev, 
                                       n = n,
                                       preprocess = preprocess,
                                       only_allow_annotators=only_allow_annotators)
            split_dfs.append(('dev', dev))
        
        if path_test:
            test = cls._load_preprocessed(path_test, 
                                       n = n,
                                       preprocess = preprocess,
                                       only_allow_annotators=only_allow_annotators)
            split_dfs.append(('test', test))

        annotator_ids = set(annotator_id for _, df in split_dfs for annotator_id in df['annotator_ids'].explode())
        annotators_mapping = {annotator: index for index, annotator in enumerate(annotator_ids)}

        splits = datasets.DatasetDict()
        for name, data in split_dfs:
            # never aggregate evaluation labels
            do_aggregate = name == 'train' and do_majority_aggregation
            data = cls._group_labels(data, do_aggregate)
            data['annotator_indecies'] = data['annotator_ids'].apply(lambda x: [annotators_mapping[a] for a in x])
            if not do_aggregate:
                data = cls._pad_labels_annotators(data)
            split = datasets.Split.TRAIN if name == 'train' else datasets.Split.TEST if name == 'test' else datasets.Split.VALIDATION
            dataset = datasets.Dataset.from_pandas(data, split = split)
            dataset = cls._convert_to_features(dataset, tokenizer=tokenizer, max_length=max_length)
            splits[name] = dataset
    
        return cls(splits, annotators_mapping, None)

    @classmethod
    def _load_raw(
            cls,
            path,
            n = None,
            only_allow_annotators=None
        ):
        df = pd.read_csv(path, sep=',', nrows = n)
        return pd.DataFrame.from_records([{
            'id': row['unit_id'],
            'text': row['text'],
            'annotator_ids': annotator,
            'labels': int(row[f'concreteness_{annotator}'])
        } for _, row in df.iterrows() for annotator in range(1,6) if not only_allow_annotators or annotator in only_allow_annotators])
    
    @classmethod
    def _load_preprocessed(
            cls,
            path,
            preprocess = None,
            n = None,
            only_allow_annotators=None
        ):
        df = cls._load_raw(path, n = n,only_allow_annotators=only_allow_annotators)
        if callable(preprocess):
            df['text'] = df['text'].apply(preprocess)
        return df

    @classmethod
    def _group_labels(
        cls,
        df,
        do_majority_aggregation: bool = False
    ):  
        def majority(labels):
            l = labels.to_numpy()
            missing_labels_val = -1
            return scipy.stats.mode(l[l > missing_labels_val], nan_policy='omit').mode.flatten().astype(int).tolist()
        
        annotations = df[['id', 'text', 'annotator_ids', 'labels']] \
                        .groupby(['id']) \
                        .agg({
                            'text': 'first', 
                            'annotator_ids': list, 
                            'labels': (lambda lables: majority(lables)) if do_majority_aggregation else list
                        })
        return annotations

    @classmethod
    def _convert_to_features(cls, dataset, tokenizer, max_length):
        features = dataset.map(
            FeatureConversion(
                tokenizer,
                max_length
            ),
            batched=True,
            load_from_cache_file=False,
        )

        column_names = features.column_names
        to_remove_columns = [column for column in column_names if column not in ['input_ids', 'attention_mask', 'labels', 'annotator_indecies']]
        features = features.remove_columns(to_remove_columns)
        
        features.set_format(
            type="torch", 
            columns=['input_ids', 'attention_mask', 'labels', 'annotator_indecies']
        )
        return features

class FeatureConversion(object):

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, example_batch):
            inputs = example_batch['text']
            features = self.tokenizer(
                inputs, 
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True
            )
            return features