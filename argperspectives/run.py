from functools import reduce
import json
import operator
from typing import Union
import scipy
import numpy as np
from sklearn.utils import compute_class_weight
import torch
import transformers
from transformers import AutoModelForSequenceClassification

from transformers.trainer import Trainer

from transformers.data import DefaultDataCollator
import datasets
from argperspectives.datasets.splits import create_splits

from argperspectives.eval.callbacks import EvalResultsCallback
from argperspectives.eval.loop import eval_singletask
from argperspectives.eval.metrics import MultiAnnotatorMetrics
from argperspectives.eval.metrics import MajorityLabelMetrics
from enum import Enum
import logging

from argperspectives.training.recommender import AnnotatorSetDataCollator
logging.basicConfig(level=logging.INFO)

from argperspectives.models import RecommenderComponents

from pathlib import Path

from argperspectives.models import PerAnnotatorModelForSequenceClassification
from argperspectives.models import Recommender
from argperspectives.models import RobertaForSequenceMultiTaskClassification

from argperspectives.datasets import (
    cimt, 
    valnov
)


from argperspectives.training import WeightedTrainer
from argperspectives.training import compute_label_weights


# inherit from str for easy serialization to JSON
class Architecture(str, Enum):
    SINGLE_TASK = 'single_task'
    MULTI_TASK = 'multi_task'
    RECOMMENDER = 'recommender'

# inherit from str for easy serialization to JSON
class EvalSetting(str, Enum):
    TRAIN_TEST_SPLIT = 'TRAIN_TEST_SPLIT'
    K_FOLD = 'K_FOLD'

# inherit from str for easy serialization to JSON
class DatasetType(str, Enum):
    VALNOV = 'VALNOV'
    CIMT = 'CIMT'

# inherit from str for easy serialization to JSON
class ClassifierType(str, Enum):
    LINEAR_LAYER = 'LINEAR_LAYER'
    HEAD = 'HEAD'

def init_multiannotator_model(
        classifier_type,
        pretrained_name_path,
        annotators_mapping,
        label_weights,
        num_labels
    ):
        if classifier_type == ClassifierType.HEAD:
                    #num_labels = list(label_weights.values())[0].shape[0]
                    model = PerAnnotatorModelForSequenceClassification(
                        pretrained_name_path,
                        annotators_mapping=annotators_mapping,
                        label_weights=label_weights,
                        num_labels=num_labels
                    )
        elif classifier_type == ClassifierType.LINEAR_LAYER:
            raise NotImplementedError()
        else:
            raise ValueError(f'classifier_type needs to be ClassifierType.LINEAR_LAYER or ClassifierType.HEAD, got {classifier_type}')
        return model
def init_recommender(
        recommender_params,
        pretrained_name_path, 
        all_annotator_ids,
        label_weights,
        num_labels
    ):
    if recommender_params and 'user_tower' in recommender_params:
        user_tower_class = recommender_params['user_tower']['class']
        if user_tower_class == 'LinearEncoder':
            params = recommender_params['user_tower']
            if 'use_activation_function' in params and params['use_activation_function']:
                activation_function = torch.nn.ReLU()
            else:
                activation_function = None
            user_tower = RecommenderComponents.LinearEncoder(
                all_user_ids = all_annotator_ids,
                embedding_size = params['embedding_size'] if 'embedding_size' in params else 25,
                num_layers = params['num_layers'] if 'num_layers' in params else 1,
                activation_function = activation_function,
                dropout = .2
            )
        else:
            raise ValueError(f'If given, user tower class needs to be "LinearEnconder", got {user_tower_class}')
    else:
        user_tower = RecommenderComponents.OneHotEncoder(all_user_ids = all_annotator_ids)

    if recommender_params and 'combiner' in recommender_params:
        embedding_size = 25
        if 'user_tower' in recommender_params and 'embedding_size' in recommender_params['user_tower']:
            embedding_size = recommender_params['user_tower']['embedding_size']
        params = recommender_params['combiner']
        combiner_class = params['class']
        if combiner_class == 'LinearCombiner':
            activation_function = None
            if 'activation_function' in params:
                if params['activation_function'].lower() == 'relu':
                    activation_function = torch.nn.ReLU()
                elif params['activation_function'].lower() == 'tanh':
                    activation_function = torch.nn.Tanh()
            combiner = RecommenderComponents.LinearCombiner(
                user_embedding_length= embedding_size,
                output_shape = num_labels,
                num_layers = params['num_layers'] if 'num_layers' in params else 1,
                activation_function = activation_function,
                dropout = 0.2
            )
        elif combiner_class == 'DeepCrossNetwork':
            combiner = RecommenderComponents.DeepCrossNetwork(
                user_embedding_length= embedding_size,
                output_shape = num_labels,
                activation_function = torch.nn.ReLU(),
                form = params['form'],
                num_layers = params['num_layers'] if 'num_layers' in params else 3,
                dense_feature_appendix = params['dense_feature_appendix'] if 'dense_feature_appendix' in params else None
            )
        else:
            raise ValueError(f'If given, combiner class needs to be "LinearCombiner" or "DeepCrossNetwork", got {user_tower_class}')
    else:
        combiner = RecommenderComponents.MatMulCombiner(
                                        user_embedding_length=len(all_annotator_ids),
                                        output_shape = num_labels
                                    )
    
    similar_parameter_strength = None
    shared_private_model = None
    if recommender_params and 'text_tower' in recommender_params:
        if 'similar_parameter_strength' in recommender_params['text_tower']:
            similar_parameter_strength = recommender_params['text_tower']['similar_parameter_strength']
        if 'shared_private_model' in recommender_params['text_tower']:
            shared_private_model = recommender_params['text_tower']['shared_private_model']
    
    return Recommender(pretrained_name_path, 
                        user_tower = user_tower, 
                        combiner = combiner,
                        user_ids = all_annotator_ids,
                        class_weights = label_weights,
                        text_processing_transformer_tower_similar_parameter_strength = similar_parameter_strength,
                        shared_private_model = shared_private_model)

def train(
        experiment_path,
        experiment_index,
        data_path,
        dev_path = None,
        test_path = None,
        dataset_type = DatasetType.VALNOV,
        pretrained_name_path='distilbert-base-uncased',
        architecture=Architecture.MULTI_TASK,
        model_for_annotator: Union[int, str] = None,
        classifier_type=ClassifierType.HEAD,
        recommender_params=None,
        learning_rate=1e-7,
        num_train_epochs=3,
        batch_size=2,
        n=None,
        eval_setting=EvalSetting.TRAIN_TEST_SPLIT,
        k=4,
        eval_while_train=False,
        output_dir='.models/',
        random_seeds = [2803636207]
    ) -> None:
    if torch.cuda.is_available():
        logging.info('CUDA is available')
    else:
        logging.warning('Training on CPU! CUDA not available')

    architecture = Architecture[architecture]
    eval_setting = EvalSetting[eval_setting]
    dataset_type = DatasetType[dataset_type]
    classifier_type = ClassifierType[classifier_type]

    config = {
        'experiment_index': experiment_index,
        'experiment_path': str(experiment_path),
        'data_path': str(data_path),
        'dev_path': str(dev_path),
        'test_path': str(test_path),
        'dataset_type': dataset_type,
        'pretrained_name_path': pretrained_name_path,
        'architecture': architecture,
        'model_for_annotator': model_for_annotator,
        'classifier_type': classifier_type,
        'learning_rate': learning_rate,
        'num_train_epochs': num_train_epochs,
        'batch_size': batch_size,
        'n': n,
        'k': k,
        'eval_setting': eval_setting,
        'output_dir': output_dir,
        'eval_while_train': eval_while_train,
        'recommender_params': str(recommender_params)
    }

    if dataset_type == DatasetType.VALNOV:
        all_data = valnov.Dataset.load(
            data_path,
            dev_path,
            test_path,
            n=n,
            do_majority_aggregation = architecture == Architecture.SINGLE_TASK,
            do_pad_to_num_annotators = architecture == Architecture.RECOMMENDER,
            model_name = pretrained_name_path,
            only_allow_annotators = [model_for_annotator] if model_for_annotator else None
        )
        classes = [0,1]
        num_labels=(2,2)

    elif dataset_type == DatasetType.CIMT:
        all_data = cimt.Dataset.load(
            data_path,
            dev_path,
            test_path,
            n=n,
            do_majority_aggregation = architecture == Architecture.SINGLE_TASK,
            model_name = pretrained_name_path,
            only_allow_annotators = [model_for_annotator] if model_for_annotator else None
        )
        classes = [0,1,2]
        num_labels = 3
    else:
        raise ValueError(f'dataset_type needs to be DatasetType.VALNOV, DatasetType.CIMT or DatasetType.LEWIDI, got {dataset_type}')
    
    dataset = all_data.features

    # if type(dataset) == datasets.dataset_dict.DatasetDict:
    #     classes = dataset['train']['labels'].flatten().unique().tolist()
    # else:
    #     classes = dataset['labels'].flatten().unique().tolist()
    #np.unique(dataset['train']['labels'].flatten(end_dim=-2).numpy(), axis = 0)
    #classes = [c for c in classes if c > -100]
    #num_labels = len(classes) if dataset['train']['labels'].dim() < 3 else (len(classes), dataset['train']['labels'].shape[-1])
    
    metrics = MultiAnnotatorMetrics(
        all_data.annotators_mapping,
        classes,
        is_annotator_set_batches = architecture == Architecture.RECOMMENDER,
        label_names = ('validity', 'novelty') if dataset_type == DatasetType.VALNOV else None
    )

    for random_seed in random_seeds:
        if type(all_data.features) == datasets.dataset_dict.DatasetDict:
            # if dataset is split already, use splits
            splits = [
                all_data.features
            ]
        else:
            splits = create_splits(
                dataset, 
                k=k,
                random_state=random_seed,
                create_validation_set = eval_while_train
            )
        for index, split in enumerate(splits):

            config['random_seed'] = random_seed
            config['split'] = index
            
            common_training_args = {
                'output_dir': output_dir,
                'overwrite_output_dir': True,
                'learning_rate': learning_rate,
                'do_train': True,
                'num_train_epochs': num_train_epochs,
                'per_device_train_batch_size': batch_size,
                'per_device_eval_batch_size': batch_size,
                'evaluation_strategy': 'epoch' if 'dev' in split and architecture != Architecture.SINGLE_TASK else 'no',
                'save_strategy': 'no', #FIXME this is a workaround because at least with Recommender saving the trainer state crashes
                'logging_steps': 100,
                'save_total_limit': 1,
                # seed (int, optional, defaults to 42) — Random seed that will be set at the beginning of training. 
                # To ensure reproducibility across runs, use the model_init function to instantiate 
                # the model if it has some randomly initialized parameters.
                # https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/trainer#transformers.TrainingArguments.seed
                # data_seed (int, optional) — Random seed to be used with data samplers. If not set, random generators for data 
                # sampling will use the same seed as seed. This can be used to ensure reproducibility of data sampling, 
                # independent of the model seed.
                # https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/trainer#transformers.TrainingArguments.data_seed
                'seed': random_seed
            }

            if architecture != Architecture.SINGLE_TASK:

                label_weights = compute_label_weights(
                    labels=split['train']['labels'],
                    annotators_on_example=split['train']['annotator_indecies'],
                    classes=classes
                )

                training_args = transformers.TrainingArguments(
                    **common_training_args,
                    label_names = ['labels', 'annotator_indecies']
                )

                if architecture == Architecture.RECOMMENDER:
                    all_annotator_ids = tuple(all_data.annotators_mapping.values())
                    init_func = lambda: init_recommender(recommender_params,
                                                         pretrained_name_path, 
                                                         all_annotator_ids, 
                                                         label_weights, 
                                                         num_labels
                                                    )
                elif architecture == Architecture.MULTI_TASK: 
                    init_func = lambda: init_multiannotator_model(classifier_type, pretrained_name_path, all_data.annotators_mapping, label_weights, num_labels)
                else:
                    raise ValueError('Value of argument "architecture" needs to be Architecture.SINGLE_TASK, Architecture.MULTI_TASK or Architecture.RECOMMENDER')

                trainer = Trainer(
                        model_init = init_func,
                        args=training_args,
                        train_dataset=split['train'],
                        eval_dataset=split['dev'] if 'dev' in split else None,
                        compute_metrics = metrics.compute,
                        data_collator = AnnotatorSetDataCollator() if architecture == Architecture.RECOMMENDER else DefaultDataCollator(),
                        callbacks=[
                            EvalResultsCallback(
                                experiment_path=experiment_path,
                                experiment_config = config
                            )
                        ]
                )
                logging.info(f'Running on split {index}')
                trainer.train()
                if 'test' in split:
                    trainer.evaluate(eval_dataset=split['test'])

            elif architecture == Architecture.SINGLE_TASK:
                if classifier_type == ClassifierType.LINEAR_LAYER:
                    raise NotImplementedError("We only implement classification heads as they are provided from transformers' RobertaClassificationHead")
                elif classifier_type == ClassifierType.HEAD:
                    model_init = lambda: RobertaForSequenceMultiTaskClassification.from_pretrained(
                        pretrained_name_path,
                        num_labels = num_labels if isinstance(num_labels, int) else reduce(operator.mul, num_labels, 1),
                        problem_type = 'single_label_classification'
                    )
                else:
                    raise ValueError(f'classifier_type needs to be ClassifierType.LINEAR_LAYER or ClassifierType.HEAD, got {classifier_type}')

                if 'train' in split:
                    if 'annotator_indecies' in split['train'].column_names:
                        split['train'] = split['train'].remove_columns(['annotator_indecies'])

                labels = split['train']['labels'].numpy() # num_examples x 1
                labels = labels.reshape(labels.shape[0]) if labels.shape[1] == 1 else labels  # num_examples or num_examples x num_labels (for ValNov)
                MISSING_LABELS_VALUE = -100
                if labels.ndim == 1:
                    label_weights_np = compute_class_weight(
                                        'balanced',
                                        classes=classes, 
                                        y=labels[labels > MISSING_LABELS_VALUE]

                                )
                elif labels.ndim == 2:
                    label_weights_np = np.stack([compute_class_weight(
                            'balanced',
                            classes=classes, 
                            y=l[l > MISSING_LABELS_VALUE]
                    ) for l in labels.T])
                else:
                    raise ValueError(f'Expected labels to be of dimensionality 1 or 2, got {labels.ndim}')
                
                label_weights = torch.tensor(label_weights_np, dtype=torch.float32)

                training_args = transformers.TrainingArguments(
                    **common_training_args
                )

                trainer = WeightedTrainer(
                    model_init=model_init,
                    args=training_args,
                    train_dataset=split['train'],
                    label_weights = label_weights
                )

                logging.info(f'Running on split {index}')
                trainer.train()
                if 'dev' in split:
                    eval_singletask(
                        test_data = split['dev'], 
                        metrics = metrics, 
                        trainer = trainer,
                        experiment_config = config,
                        experiment_path = experiment_path
                    )
                if 'test' in split:
                    eval_singletask(
                        test_data = split['test'], 
                        metrics = metrics, 
                        trainer = trainer,
                        experiment_config = config,
                        experiment_path = experiment_path
                    )
        
            if eval_setting == EvalSetting.TRAIN_TEST_SPLIT:
                 # Add break to make stratified k-fold effectively stratified train/test split
                break
            elif eval_setting == EvalSetting.K_FOLD:
                pass
            else:
                raise ValueError(f'eval_setting needs to be EvalSetting.TRAIN_TEST_SPLIT or EvalSetting.K_FOLD, got {eval_setting}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train annotator-aware models.')
    parser.add_argument('experiment_path', help='a path to directory with an experiment configuration config.json')

    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)

    with open(experiment_path / 'config.json') as f:
        config = json.load(f)

    for index, experiment in enumerate(config['settings']):
        train(
            experiment_path,
            experiment_index = index,
            **experiment,
            output_dir="./models/"
        )
