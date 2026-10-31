# Document Classification with Random Forest on Apache Spark

A PySpark pipeline that sorts posts from the 20 Newsgroups dataset into one of 20 categories. It turns raw text into TF-IDF vectors, trains a Random Forest classifier, tunes its hyperparameters on a validation split, and evaluates the final model on the official test set. Everything runs on Spark's DataFrame-based MLlib API.

## Dataset

The project uses the "bydate" version of [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/), with its predefined train/test partition. The dataset is not included in this repository.

```
20news-bydate-train/   11,314 posts   (split 80/20 into train / validation, seed 42)
20news-bydate-test/     7,532 posts   (used only for the final evaluation)
  alt.atheism/
  comp.graphics/
  ...
  talk.religion.misc/   one file per post, 20 category folders
```

Labels 0 to 19 follow the alphabetical order of the category folders.

## Pipeline

```
text -> RegexTokenizer -> StopWordsRemover -> HashingTF -> IDF -> RandomForestClassifier
```

| Stage | Settings | Role |
| --- | --- | --- |
| RegexTokenizer | lowercase, split on `\W`, min token length 2 | text to tokens |
| StopWordsRemover | default English list | drops common words |
| HashingTF | 4096 features, raw counts | tokens to a sparse term-count vector |
| IDF | minDocFreq 2, fit on training data only | weights rare terms up, common terms down |
| RandomForestClassifier | featureSubsetStrategy `sqrt`, maxBins 32, seed 42 | multi-class classifier |

## Steps

1. **Load and split.** Read both partitions into a DataFrame (`doc_id`, `text`, `label`, `category`, `source_split`). Split the official training set 80/20 into train and validation, check that there's no overlap and all 20 classes are present, and save the split assignments.
2. **Feature extraction.** Fit the TF-IDF stages on the training subset only, so no validation or test data leaks into the IDF weights.
3. **Baselines.** A majority-class baseline and an initial Random Forest (20 trees, depth 5).
4. **Tuning.** A `ParamGridBuilder` grid of 30 configurations (`numTrees` 1/5/20/50/100, `maxDepth` 5/10/25, `minInstancesPerNode` 1/5). Each one is scored on the validation split, and the best weighted F1 is selected.
5. **Final evaluation.** Refit the selected pipeline on the full official training set, then report accuracy, weighted precision, recall and F1, macro F1, per-class metrics and a 20 x 20 confusion matrix on the test set.

## Running

```bash
# put the dataset on HDFS (cluster) or point DATA at a local path
hdfs dfs -put 20news-bydate-train 20news-bydate-test /PA2/

spark-submit --master spark://phoenix.cs.colostate.edu:7077 PA2.py --data /user/C837364858/PA2
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--data` | `/user/C837364858/PA2` | HDFS folder holding the train and test folders (`file:///full/path` in local mode) |
| `--train-dir`, `--test-dir` | `20news-bydate-train`, `20news-bydate-test` | names of the two split folders, for a new dataset |
| `--out` | `PA2_output` | HDFS folder for the saved document-to-split assignments |
| `--results` | `results` | local folder for the CSV result tables used in the report |
| `--partitions` | `8` | partitions for the document DataFrames |

The category list and labels are read from the training folders, so a dataset with different categories works without code changes.

## Requirements

- Apache Spark 3.5 with PySpark
- Python 3
