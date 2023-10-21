
import datetime
import datasets
import pandas as pd
from transformers.trainer import Trainer
from transformers import EvalPrediction
from argperspectives.eval.metrics import MultiAnnotatorMetrics

def eval_singletask(
    test_data: datasets.Dataset,
    metrics: MultiAnnotatorMetrics,
    trainer: Trainer,
    experiment_config: dict,
    experiment_path: str,
):
    """ Function to evaluate a majority / singletask model against majority and individual labels """
    to_remove_columns = [c for c in ['annotator_indecies', 'labels'] if c in test_data.column_names]
    prediction_input = test_data.remove_columns(to_remove_columns)
    output = trainer.predict(test_dataset=prediction_input)
    logits = output.predictions
    result = metrics.compute_single_predictions(EvalPrediction(logits, (test_data['labels'].numpy(), test_data['annotator_indecies'].numpy())))
    result['timestamp'] = str(datetime.datetime.now())
    result.update(experiment_config)
    with open(f'{experiment_path}/result.csv', 'a') as f:
        df = pd.DataFrame([result])
        df['evaluation_set'] = 'test' if test_data.split == datasets.Split.TEST else 'dev'
        df = df.set_index('timestamp')
        df.to_csv(f, header=not f.tell())