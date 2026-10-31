"""
CS435 PA2: Document Classification with Random Forest on Apache Spark

text -> RegexTokenizer -> StopWordsRemover -> HashingTF -> IDF -> RandomForestClassifier

Run on the cluster:
    spark-submit --master spark://phoenix.cs.colostate.edu:7077 PA2.py --data /user/C837364858/PA2

Run on a new dataset (same <name>-train / <name>-test layout, one folder per category):
    spark-submit ... PA2.py --data /user/C837364858/NEWDATA --train-dir new-train --test-dir new-test

Small result tables are written as CSV files to --results (a local folder on the driver) for the report.
The document-to-split assignments are written to --out on HDFS.
"""
import argparse
import csv
import math
import os
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IDF, HashingTF, RegexTokenizer, StopWordsRemover
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/user/C837364858/PA2", help="HDFS folder holding the train and test folders")
parser.add_argument("--train-dir", default="20news-bydate-train")
parser.add_argument("--test-dir", default="20news-bydate-test")
parser.add_argument("--out", default="PA2_output", help="HDFS folder for the saved split assignments")
parser.add_argument("--results", default="results", help="local folder for the CSV result tables")
parser.add_argument("--partitions", type=int, default=8, help="partitions for the document DataFrames")
args = parser.parse_args()

SEED = 42
os.makedirs(args.results, exist_ok=True)

spark = SparkSession.builder.appName("PA2").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


def section(title):
    print("\n" + "=" * 80 + "\n" + title + "\n" + "=" * 80)


# Writes a small, already-collected table to a local CSV for the report
def write_csv(name, header, rows):
    with open(os.path.join(args.results, name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


'''
Task 1: Load, inspect, and partition the data

Each split is a folder with one sub-folder per category and one file per post:
--20news-bydate-train
  --alt.atheism
    *49960*
  ...
--20news-bydate-test
  ...

Target schema:
------------------------------------------------------
| doc_id | text | label | category | source_split |
------------------------------------------------------
'''
section("Task 1: Load, inspect, and partition the data")


def load(split, folder):
    raw = spark.read.text(f"{args.data}/{folder}/*/*", wholetext=True).withColumn("path", F.input_file_name())
    return (raw
            .withColumn("category", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 1))   # folder name = category
            .withColumn("file_name", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 2))  # file name = post number
            # The same post number can appear in both splits, so the split is part of the ID
            .withColumn("doc_id", F.concat_ws("/", F.lit(split), "category", "file_name"))
            .withColumn("source_split", F.lit(split))
            # Cleaning: drop the e-mail header block (everything up to the first blank line).
            # Headers are metadata (sender, host, newsgroup names), not article text.
            # A post with no blank line is kept unchanged.
            .withColumn("text", F.regexp_replace("value", r"^(?s).*?\n\s*\n", ""))
            .select("doc_id", "text", "category", "source_split"))


train_raw = load("train", args.train_dir)
test_raw = load("test", args.test_dir)

# The label mapping comes from the training folders only, in alphabetical order (like scikit-learn's loader).
# It's 20 names, so collecting them to the driver is fine.
CATEGORIES = sorted(row["category"] for row in train_raw.select("category").distinct().collect())
print(len(CATEGORIES), "categories:", CATEGORIES)

labels = spark.createDataFrame(
    [(category, float(i)) for i, category in enumerate(CATEGORIES)],
    "category string, label double")

# Broadcasts the 20-row label table to every executor, so the join doesn't shuffle the corpus.
# Reading ~11k small files gives hundreds of tiny partitions; repartitioning to a few keeps the
# Random Forest from aggregating split statistics across hundreds of tasks on every tree level.
official_train = (train_raw.join(F.broadcast(labels), "category")
                  .select("doc_id", "text", "label", "category", "source_split")
                  .repartition(args.partitions, "doc_id")
                  .cache())
official_test = (test_raw.join(F.broadcast(labels), "category")
                 .select("doc_id", "text", "label", "category", "source_split")
                 .repartition(args.partitions, "doc_id")
                 .cache())

official_train.printSchema()
n_official_train, n_official_test = official_train.count(), official_test.count()
print("official train:", n_official_train, "| official test:", n_official_test, "| total:", n_official_train + n_official_test)

# Empty documents are kept (they get zero feature vectors), only counted
tokenize = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W", gaps=True, minTokenLength=2)
remove_stop_words = StopWordsRemover(inputCol="words", outputCol="kept")
empty_rows = []
for name, df in [("train", official_train), ("test", official_test)]:
    tokens = remove_stop_words.transform(tokenize.transform(df))
    counts = tokens.agg(
        F.sum((F.trim("text") == "").cast("int")).alias("empty_text"),
        F.sum((F.size("words") == 0).cast("int")).alias("no_tokens"),
        F.sum((F.size("kept") == 0).cast("int")).alias("no_tokens_after_stop_words")).first()
    empty_rows.append([name, counts["empty_text"], counts["no_tokens"], counts["no_tokens_after_stop_words"]])
    print(f"{name}: empty after cleaning {counts['empty_text']} | no tokens {counts['no_tokens']} "
          f"| no tokens after stop-word removal {counts['no_tokens_after_stop_words']}")
write_csv("task1_empty_documents.csv", ["split", "empty_text", "no_tokens", "no_tokens_after_stop_words"], empty_rows)

# Split only the official training set: ~80% train, ~20% validation.
# Only doc_id is repartitioned, so the split doesn't move the text around.
# A fixed partition layout + seed makes randomSplit reproducible.
ids = official_train.select("doc_id").repartition(args.partitions, "doc_id").sortWithinPartitions("doc_id")
train_ids, validation_ids = ids.randomSplit([0.8, 0.2], seed=SEED)


# Turns a DataFrame of doc_ids back into full rows, laid out the same way on every run
def rows_for(id_df):
    return official_train.join(id_df, "doc_id").repartition(args.partitions, "doc_id").sortWithinPartitions("doc_id").cache()


train, validation = rows_for(train_ids), rows_for(validation_ids)
n_train, n_validation = train.count(), validation.count()
print("train:", n_train, "| validation:", n_validation, "| test:", n_official_test)
print("train/validation overlap:", train.select("doc_id").intersect(validation.select("doc_id")).count())

# randomSplit isn't stratified, so check every class is in both parts
distribution = (train.groupBy("label", "category").count().withColumnRenamed("count", "train")
                .join(validation.groupBy("label", "category").count().withColumnRenamed("count", "validation"),
                      ["label", "category"], "full_outer")
                .join(official_test.groupBy("label", "category").count().withColumnRenamed("count", "test"),
                      ["label", "category"], "full_outer")
                .fillna(0)
                .orderBy("label"))
distribution.show(len(CATEGORIES), truncate=False)
distribution_rows = distribution.collect()
print("classes in train:", sum(r["train"] > 0 for r in distribution_rows),
      "| classes in validation:", sum(r["validation"] > 0 for r in distribution_rows))
write_csv("task1_class_distribution.csv", ["label", "category", "train", "validation", "test"],
          [[int(r["label"]), r["category"], r["train"], r["validation"], r["test"]] for r in distribution_rows])
write_csv("task1_split_sizes.csv", ["split", "documents"],
          [["train", n_train], ["validation", n_validation], ["test", n_official_test]])

# Saves the document-to-split assignments for reproducibility
assignments = (train.select("doc_id", F.lit("train").alias("assigned_split"))
               .unionByName(validation.select("doc_id", F.lit("validation").alias("assigned_split")))
               .unionByName(official_test.select("doc_id", F.lit("test").alias("assigned_split"))))
assignments.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{args.out}/split_assignments")
print("split assignments saved to", f"{args.out}/split_assignments")


'''
Task 2: TF-IDF feature extraction

RegexTokenizer, StopWordsRemover and HashingTF are Transformers (fixed rules).
IDF is an Estimator: fit() counts document frequencies, so it only ever sees training documents.
'''
section("Task 2: TF-IDF feature extraction")


def feature_stages():
    return [
        RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W", gaps=True, minTokenLength=2, toLowercase=True),
        StopWordsRemover(inputCol="words", outputCol="kept"),
        HashingTF(inputCol="kept", outputCol="counts", numFeatures=4096, binary=False),
        IDF(inputCol="counts", outputCol="features", minDocFreq=2),
    ]


# Fit on the training subset only. Pipeline.fit() returns a PipelineModel; it is not fitted twice.
feature_model = Pipeline(stages=feature_stages()).fit(train)
print("fitted stages:", [type(stage).__name__ for stage in feature_model.stages])

# Inspect the training document with the smallest ID that has tokens
row = (feature_model.transform(train.orderBy("doc_id").limit(50))
       .filter(F.size("kept") > 0)
       .orderBy("doc_id")
       .first())
print("document:", row["doc_id"], "| category:", row["category"])
print("excerpt:", repr(row["text"][:300]))
print("tokens:", len(row["words"]), row["words"][:25])
print("after stop-word removal:", len(row["kept"]), row["kept"][:25])
print("vector size:", row["features"].size, "| nonzero entries:", row["features"].numNonzeros())

# Manual TF-IDF check for the document's most frequent term.
# HashingTF has only 4096 slots for a vocabulary of tens of thousands of words, so many words share a
# slot (hash collisions). Spark counts TF and DF per slot, not per word. The check uses the slot's counts,
# and also prints the word-only DF (the "no collisions" assumption) to show what a collision changes.
hashing_tf, idf_model = feature_model.stages[2], feature_model.stages[3]
term = max(sorted(set(row["kept"])), key=row["kept"].count)
index = hashing_tf.indexOf(term)
tf_word, tf_slot = row["kept"].count(term), row["counts"][index]
df_word = (feature_model.stages[1].transform(feature_model.stages[0].transform(train))
           .filter(F.array_contains("kept", term)).count())
df_slot = idf_model.docFreq[index]
word_idf = math.log((n_train + 1) / (df_word + 1))
manual_idf = math.log((n_train + 1) / (df_slot + 1))
print(f"manual check, term '{term}' -> slot {index}, |D|={n_train}")
print(f"  word only (no collisions): TF={tf_word}, DF={df_word}, IDF={word_idf:.6f}, TF-IDF={tf_word * word_idf:.6f}")
print(f"  slot (what Spark counts):  TF={tf_slot:.0f}, DF={df_slot}, "
      f"IDF=log(({n_train}+1)/({df_slot}+1))={manual_idf:.6f}, TF-IDF={tf_slot * manual_idf:.6f}")
print(f"  Spark:                     IDF={idf_model.idf[index]:.6f}, TF-IDF={row['features'][index]:.6f}")
write_csv("task2_feature_example.csv", ["field", "value"], [
    ["doc_id", row["doc_id"]], ["excerpt", row["text"][:300]],
    ["tokens", len(row["words"])], ["tokens_after_stop_words", len(row["kept"])],
    ["vector_size", row["features"].size], ["nonzero_entries", row["features"].numNonzeros()],
    ["check_term", term], ["slot", index], ["tf_word", tf_word], ["df_word", df_word],
    ["tfidf_word_no_collisions", tf_word * word_idf], ["tf_slot", tf_slot], ["df_slot", df_slot],
    ["manual_tfidf_slot", tf_slot * manual_idf], ["spark_tfidf", row["features"][index]]])


'''
Task 3: Baselines

Majority-class baseline: predict the training subset's most common label for every validation document.
Initial Random Forest: the full pipeline, fitted on train, scored on validation.
'''
section("Task 3: Baselines")

accuracy = MulticlassClassificationEvaluator(metricName="accuracy")
weighted_f1 = MulticlassClassificationEvaluator(metricName="f1")  # weighted F1, not macro F1

majority = train.groupBy("label").count().orderBy(F.desc("count"), "label").first()["label"]
baseline = validation.withColumn("prediction", F.lit(majority))
baseline_accuracy, baseline_f1 = accuracy.evaluate(baseline), weighted_f1.evaluate(baseline)
print("majority category:", CATEGORIES[int(majority)])

forest = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20, maxDepth=5,
                                featureSubsetStrategy="sqrt", minInstancesPerNode=1, maxBins=32, seed=SEED)
start = time.time()
baseline_model = Pipeline(stages=feature_stages() + [forest]).fit(train)
baseline_seconds = time.time() - start

predictions = baseline_model.transform(validation).select("label", "prediction").cache()
forest_accuracy, forest_f1 = accuracy.evaluate(predictions), weighted_f1.evaluate(predictions)
predictions.unpersist()

baseline_rows = [["majority class (" + CATEGORIES[int(majority)] + ")", round(baseline_accuracy, 4), round(baseline_f1, 4), ""],
                 ["random forest (20 trees, depth 5)", round(forest_accuracy, 4), round(forest_f1, 4), round(baseline_seconds, 1)]]
print("%-45s %10s %12s %10s" % ("model", "accuracy", "weighted F1", "fit (s)"))
for r in baseline_rows:
    print("%-45s %10s %12s %10s" % tuple(r))
write_csv("task3_baselines.csv", ["model", "validation_accuracy", "validation_weighted_f1", "fit_seconds"], baseline_rows)


'''
Task 4: Tune Random Forest hyperparameters

The feature settings never change, so the feature pipeline is fitted once on train and the
TF-IDF vectors are cached. Only the forest is refitted for each of the 30 configurations.
'''
section("Task 4: Hyperparameter tuning")

forest = RandomForestClassifier(labelCol="label", featuresCol="features",
                                featureSubsetStrategy="sqrt", maxBins=32, seed=SEED)
grid = (ParamGridBuilder()
        .addGrid(forest.numTrees, [1, 5, 20, 50, 100])
        .addGrid(forest.maxDepth, [5, 10, 25])
        .addGrid(forest.minInstancesPerNode, [1, 5])
        .build())
print("configurations:", len(grid))

train_features = feature_model.transform(train).select("label", "features").cache()
validation_features = feature_model.transform(validation).select("label", "features").cache()
print("cached rows:", train_features.count(), validation_features.count())

tuning_header = ["numTrees", "maxDepth", "minInstancesPerNode", "accuracy", "weighted_f1", "fit_seconds"]
results = []
for i, params in enumerate(grid, 1):
    start = time.time()
    fitted = forest.fit(train_features, params)  # fit() runs the Spark jobs, so this times training
    fit_seconds = time.time() - start
    scored = fitted.transform(validation_features).select("label", "prediction").cache()
    results.append({
        "numTrees": params[forest.numTrees],
        "maxDepth": params[forest.maxDepth],
        "minInstancesPerNode": params[forest.minInstancesPerNode],
        "accuracy": accuracy.evaluate(scored),
        "weighted_f1": weighted_f1.evaluate(scored),
        "fit_seconds": fit_seconds,
    })
    scored.unpersist()
    r = results[-1]
    print("[%2d/%d] numTrees %3d  maxDepth %2d  minInstancesPerNode %d  accuracy %.4f  weighted F1 %.4f  fit %.1fs"
          % (i, len(grid), r["numTrees"], r["maxDepth"], r["minInstancesPerNode"], r["accuracy"], r["weighted_f1"], r["fit_seconds"]))
    # Saved after every configuration, so a crash doesn't lose finished results
    write_csv("task4_tuning_results.csv", tuning_header, [[x[k] for k in tuning_header] for x in results])

train_features.unpersist()
validation_features.unpersist()

# Highest weighted F1 (to 4 decimals); ties go to lower maxDepth, then fewer trees, then larger minInstancesPerNode
best = min(results, key=lambda r: (-round(r["weighted_f1"], 4), r["maxDepth"], r["numTrees"], -r["minInstancesPerNode"]))
print("\nselected: numTrees %d, maxDepth %d, minInstancesPerNode %d (validation weighted F1 %.4f, accuracy %.4f, fit %.1fs)"
      % (best["numTrees"], best["maxDepth"], best["minInstancesPerNode"], best["weighted_f1"], best["accuracy"], best["fit_seconds"]))
write_csv("task4_selected.csv", tuning_header, [[best[k] for k in tuning_header]])


'''
Task 5: Final model

Rebuild the selected pipeline and fit it on the entire official training set (train + validation),
which refits IDF on that set. The official test set is used here, exactly once.
'''
section("Task 5: Final model on the official test set")

final_forest = RandomForestClassifier(labelCol="label", featuresCol="features",
                                      numTrees=best["numTrees"], maxDepth=best["maxDepth"],
                                      minInstancesPerNode=best["minInstancesPerNode"],
                                      featureSubsetStrategy="sqrt", maxBins=32, seed=SEED)
start = time.time()
final_model = Pipeline(stages=feature_stages() + [final_forest]).fit(official_train)
print("final fit: %.1fs" % (time.time() - start))

test_predictions = final_model.transform(official_test).select("doc_id", "label", "prediction").cache()

overall = {name: MulticlassClassificationEvaluator(metricName=name).evaluate(test_predictions)
           for name in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]}

# Confusion counts: one row per (true label, predicted label) pair. At most 20 x 20 rows.
confusion = test_predictions.groupBy("label", "prediction").count().cache()

# Per-class TP, support (row sum = TP + FN) and predicted count (column sum = TP + FP), all from Spark aggregations
true_positives = confusion.filter(F.col("label") == F.col("prediction")).select("label", F.col("count").alias("tp"))
support = confusion.groupBy("label").agg(F.sum("count").alias("support"))
predicted = (confusion.groupBy("prediction").agg(F.sum("count").alias("predicted"))
             .withColumnRenamed("prediction", "label"))

# A metric is 0 when its denominator is 0
per_class = (labels
             .join(true_positives, "label", "left")
             .join(support, "label", "left")
             .join(predicted, "label", "left")
             .fillna(0, ["tp", "support", "predicted"])
             .withColumn("precision", F.when(F.col("predicted") > 0, F.col("tp") / F.col("predicted")).otherwise(0.0))
             .withColumn("recall", F.when(F.col("support") > 0, F.col("tp") / F.col("support")).otherwise(0.0))
             .withColumn("f1", F.when(F.col("precision") + F.col("recall") > 0,
                                      2 * F.col("precision") * F.col("recall") / (F.col("precision") + F.col("recall")))
                         .otherwise(0.0))
             .orderBy("label")
             .cache())

summary = per_class.agg(
    F.avg("f1").alias("macro_f1"),
    (F.sum(F.col("precision") * F.col("support")) / F.sum("support")).alias("weighted_precision"),
    (F.sum(F.col("recall") * F.col("support")) / F.sum("support")).alias("weighted_recall"),
    (F.sum(F.col("f1") * F.col("support")) / F.sum("support")).alias("weighted_f1")).first()

print("accuracy:            %.4f" % overall["accuracy"])
print("weighted precision:  %.4f  (from per-class table: %.4f)" % (overall["weightedPrecision"], summary["weighted_precision"]))
print("weighted recall:     %.4f  (from per-class table: %.4f)" % (overall["weightedRecall"], summary["weighted_recall"]))
print("weighted F1:         %.4f  (from per-class table: %.4f)" % (overall["f1"], summary["weighted_f1"]))
print("macro F1:            %.4f" % summary["macro_f1"])
write_csv("task5_overall_metrics.csv", ["metric", "value"], [
    ["accuracy", overall["accuracy"]], ["weighted_precision", overall["weightedPrecision"]],
    ["weighted_recall", overall["weightedRecall"]], ["weighted_f1", overall["f1"]], ["macro_f1", summary["macro_f1"]]])

per_class.select("label", "category", "support", "tp", "predicted",
                 F.round("precision", 4).alias("precision"), F.round("recall", 4).alias("recall"),
                 F.round("f1", 4).alias("f1")).show(len(CATEGORIES), truncate=False)
write_csv("task5_per_class.csv", ["label", "category", "support", "tp", "predicted", "precision", "recall", "f1"],
          [[int(r["label"]), r["category"], r["support"], r["tp"], r["predicted"], r["precision"], r["recall"], r["f1"]]
           for r in per_class.collect()])

# 20 x 20 confusion matrix: rows = true label, columns = predicted label
label_ids = [float(i) for i in range(len(CATEGORIES))]
matrix = (labels.join(confusion, "label", "left")
          .groupBy("label", "category").pivot("prediction", label_ids).agg(F.first("count"))
          .fillna(0)
          .orderBy("label"))
matrix_rows = matrix.collect()
print("rows = true label, columns = predicted label")
print("%-26s" % "true \\ predicted" + "".join("%5d" % i for i in range(len(CATEGORIES))))
for r in matrix_rows:
    print("%2d %-23s" % (int(r["label"]), r["category"][:23]) + "".join("%5d" % r[str(p)] for p in label_ids))
write_csv("task5_confusion_matrix.csv", ["true \\ predicted"] + CATEGORIES,
          [[r["category"]] + [r[str(p)] for p in label_ids] for r in matrix_rows])

print("\nresult tables written to", os.path.abspath(args.results))
spark.stop()
