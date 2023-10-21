# Architectural Sweet Spots for Modeling Human Label Variation by the Example of Argument Quality: It's Best to Relate Perspectives!

## Setup

The project uses [poetry](https://python-poetry.org/docs/) to manage dependencies.

### Structure

- Datasets: A collection of datasets for argument mining-related tasks that provide perspectivist annotations.
  - here, you can find the release of the non-aggregated [ValNov](https://phhei.github.io/ArgsValidNovel/)-dataset
  - also, you can find the used split of the [Concreteness](https://aclanthology.org/2022.argmining-1.11/)-dataset
- ``argperspectives``: The main python package containing all essential code
  - [datasets](argperspectives%2Fdatasets): Data loading and preprocessing
  - [eval](argperspectives%2Feval): Evaluation code
  - [models](argperspectives%2Fmodels): the core of our different architectures
    - [multiannotator.py](argperspectives%2Fmodels%2Fmultiannotator.py): SepHeads-architecture
    - [singletask.py](argperspectives%2Fmodels%2Fsingletask.py): PerAnnotator/MajorityVote-architecture
    - [Recommender.py](argperspectives%2Fmodels%2FRecommender.py) + [RecommenderComponents.py](argperspectives%2Fmodels%2FRecommenderComponents.py): ShareREC+SepREC-architecture
  - [training](argperspectives%2Ftraining): Training routines for the different architectures

## Usage

The entry point is the [run.py](argperspectives%2Frun.py). This code requires a config.json file. You can find example configs in the [experiments](experiments) folder.

## Citation

If you use this code, please cite our paper:

```
@inproceedings{heinisch-etal-2023-architectural,
    title = "Architectural Sweet Spots for Modeling Human Label Variation by the Example of Argument Quality: It's Best to Relate Perspectives!",
    author = "Heinisch, Philipp  and
      Orlikowski, Matthias  and
      Romberg, Julia and
      Cimiano, Philipp",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Sentosa Gateway, Singapore",
    publisher = "Association for Computational Linguistics",
}
```
