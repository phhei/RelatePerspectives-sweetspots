from pathlib import Path
from typing import Container, Iterable, Optional
import math

import json
import numpy as np

import pandas as pd
import datasets
import scipy
import transformers

class Dataset(object):
    """Loads disaggregated version of datasets from ArgMining Workshop 2022 Validity and Novelty Prediction Shared Task."""

    def __init__(self, features, annotators_mapping, sociodemographic_mapping) -> None:
        self.annotators_mapping = annotators_mapping
        self.sociodemographic_mapping = sociodemographic_mapping
        self.features = features

    @classmethod
    def _pad_labels_annotators(cls, df, pad_to_num_annotators=False):
        if pad_to_num_annotators:
            # pad to the maximum number of annotators, with annotator index being equal to index in label array
            indecies = df['annotator_indecies'].explode().unique()
            indecies.sort()
            df['labels'] = df.apply(lambda row: [row['labels'][row['annotator_indecies'].index(i)] if i in row['annotator_indecies'] else (-100, -100) for i in indecies], axis=1)
        else:
            # pad to number of maximum annotations per example
            annotations_max_length = df['annotator_indecies'].apply(len).max()
            df['annotator_indecies'] = df['annotator_indecies'].apply(lambda x: x + [-1] * (annotations_max_length - len(x)))
            df['labels'] = df['labels'].apply(lambda x: x + [(-100, -100)] * (annotations_max_length - len(x)))
        return df

    @classmethod
    def load(
            cls,
            path_train,
            path_dev = None,
            path_test = None,
            do_majority_aggregation = False,
            do_pad_to_num_annotators = False,
            preprocess = None,
            model_name : str = 'distilbert-base-uncased',
            n: Optional[int] = None,
            only_allow_annotators = None
        ):

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        max_length = tokenizer.model_max_length
        sep = f' {tokenizer._sep_token} ' if tokenizer._sep_token else ' '

        train = cls._load_preprocessed(path_train, 
                                       n = n,
                                       sep = sep,
                                       preprocess = preprocess,
                                       only_allow_annotators = only_allow_annotators)
        split_dfs = [('train', train)]
        
        if path_dev:
            dev = cls._load_preprocessed(path_dev, 
                                       n = n,
                                       sep = sep,
                                       preprocess = preprocess,
                                       only_allow_annotators = only_allow_annotators)
            split_dfs.append(('dev', dev))
        
        if path_test:
            test = cls._load_preprocessed(path_test, 
                                       n = n,
                                       sep = sep,
                                       preprocess = preprocess,
                                       only_allow_annotators = only_allow_annotators)
            split_dfs.append(('test', test))

        annotator_ids = sorted(list(set(annotator_id for _, df in split_dfs for annotator_id in df['annotator_ids'].explode())))
        annotators_mapping = {annotator: index for index, annotator in enumerate(annotator_ids)}

        splits = datasets.DatasetDict()
        for name, data in split_dfs:
            # never aggregate evaluation labels
            do_aggregate = name == 'train' and do_majority_aggregation
            data = cls._group_labels(data, do_aggregate)
            data['annotator_indecies'] = data['annotator_ids'].apply(lambda x: [annotators_mapping[a] for a in x])
            if not do_aggregate:
                data = cls._pad_labels_annotators(data, pad_to_num_annotators = do_pad_to_num_annotators)
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
            only_allow_annotators: Optional[Container[str]] = None
        ) -> pd.DataFrame:
        """ Loads seperated ValNov dataset from CSV in new format.

        Expected to contain columns for each annotator ID an their annotations for validity and novelty.
        Each column has a format like "validity_101"
        """
        df = pd.read_csv(path, sep=',', nrows = n)
        all_annotators = [col.split('_')[1] for col in df.columns if 'validity' in col]
        data = pd.DataFrame.from_records([{
            'id': row['ID'],
            'topic': row['topic'],
            'premise': row['premise'],
            'conclusion': row['conclusion'],
            'annotator_ids': annotator,
            'labels': ( 
                        0 if validity < 0 else 1 if validity > 0 else -100,
                        0 if novelty < 0 else 1 if novelty > 0 else -100
                    )
        } for _, row in df.iterrows() for annotator, validity, novelty in [(annotator, row[f'validity_{annotator}'], row[f'novelty_{annotator}']) for annotator in all_annotators] if not math.isnan(validity) or not math.isnan(novelty)])
        if only_allow_annotators:
            data = data[data['annotator_ids'].isin(only_allow_annotators)]
        return data

    @classmethod
    def _load_preprocessed(
            cls,
            path,
            sep = ' ',
            preprocess = None,
            n = None,
            only_allow_annotators = None
        ):
        df = cls._load_raw(path, n = n, only_allow_annotators = only_allow_annotators)
        if callable(preprocess):
            df['premise'] = df['premise'].apply(preprocess)
            df['conclusion'] = df['conclusion'].apply(preprocess)
        df['text'] = df['premise'] + sep + df['conclusion']
        return df

    @classmethod
    def _group_labels(
        cls,
        df,
        do_majority_aggregation: bool = False,
        missing_labels_val = -100
    ):
        def majority(labels):
            l_array = np.stack(labels.apply(lambda t: np.array(t)))
            l_tuples = tuple(l_array[:,i] for i in range(l_array.shape[1]))
            if len(labels) == 1:
                majorities = l_tuples
            else:
                majorities = [scipy.stats.mode(l[l > missing_labels_val], nan_policy='omit').mode.flatten().astype(int) for l in l_tuples]
            return np.concatenate(majorities).tolist()

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