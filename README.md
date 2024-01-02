# Architectural Sweet Spots for Modeling Human Label Variation

This project contains code and data for our EMNLP 2023 paper [Architectural Sweet Spots for Modeling Human Label Variation by the Example of Argument Quality: It's Best to Relate Perspectives!](https://aclanthology.org/2023.emnlp-main.687/) 

## Citation

The official [print](https://aclanthology.org/2023.emnlp-main.687/) is available! If you use this code, please cite our paper:

```
@inproceedings{heinisch-etal-2023-architectural,
    title = "Architectural Sweet Spots for Modeling Human Label Variation by the Example of Argument Quality: It{'}s Best to Relate Perspectives!",
    author = "Heinisch, Philipp  and
      Orlikowski, Matthias  and
      Romberg, Julia  and
      Cimiano, Philipp",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.687",
    doi = "10.18653/v1/2023.emnlp-main.687",
    pages = "11138--11154",
    abstract = "Many annotation tasks in natural language processing are highly subjective in that there can be different valid and justified perspectives on what is a proper label for a given example. This also applies to the judgment of argument quality, where the assignment of a single ground truth is often questionable. At the same time, there are generally accepted concepts behind argumentation that form a common ground. To best represent the interplay of individual and shared perspectives, we consider a continuum of approaches ranging from models that fully aggregate perspectives into a majority label to {``}share nothing{''}-architectures in which each annotator is considered in isolation from all other annotators. In between these extremes, inspired by models used in the field of recommender systems, we investigate the extent to which architectures that predict labels for single annotators but include layers that model the relations between different annotators are beneficial. By means of two tasks of argument quality classification (argument concreteness and validity/novelty of conclusions), we show that recommender architectures increase the averaged annotator-individual F1-scores up to 43{\%} over a majority-label model. Our findings indicate that approaches to subjectivity can benefit from relating individual perspectives.",
}
```

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
