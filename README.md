# XTN Multilingual Timex Detection and Normalization

This repo presents the datasets and scripts used for **multilingual timex detection and normalization** as described in

> Nayla Escribano, German Rigau and Rodrigo Agerri. 2023. [A modular approach for multilingual timex detection and normalization using deep learning and grammar-based methods](https://www.sciencedirect.com/science/article/pii/S0950705123003623?via%3Dihub). In Knowledge-Based Systems, 110612, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2023.110612.

More information on the **normalization** task can be found in the [**Spanish TimeNorm repo**](https://github.com/NGEscribano/timenorm-es/tree/master).

## Datasets

The `datasets` folder contains the datasets used for developing and evaluating multilingual timex detection and normalization.

Both datasets have been adapted to the **BIO format** to perform detection as a sequence labelling task, and only present data for timex processing (information such as events and relations have not been included).

Datasets cointained in this folder are:

- [**TimeBank**](https://catalog.ldc.upenn.edu/LDC2006T08): The **training set** was used for fine-tuning models for multilingual timex detection and for developing the [TimeNorm SCFG English](https://github.com/clulab/timenorm) and [Spanish](https://github.com/NGEscribano/timenorm-es) grammars. The **test set** has been used for evaluating both timex detection and normalization, according to the [TempEval-3 shared task](https://aclanthology.org/S13-2001/).

- [**MEANTIME**](https://aclanthology.org/L16-1699/): This corpus was used for further evaluating timex detection and normalization.

## Scripts

We include two scripts for timex processing:

- `process_timexes.py` deals with timex detection and normalizations predictions. It performs two different steps:

  - Extraction of timexes predicted by the detection model including gold data. The output file should be fed to the TimeNorm normalizer [following the corresponding instructions](https://github.com/NGEscribano/timenorm-es/blob/master/README.md).

  - Combination of output data from the detection model and the TimeNorm system and gold data. The final output file should be fed to `evaluate_timexes.py` for evaluation.

- `evaluate_timexes.py` evaluates the final prediction file.
