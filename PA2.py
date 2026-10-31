import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IDF, HashingTF, RegexTokenizer, StopWordsRemover
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Folder holding 20news-bydate-train/ and 20news-bydate-test/ (extracted from 20news-bydate.tar.gz)
DATA = sys.argv[1] if len(sys.argv) > 1 else "/PA2"           # local mode: "file:///full/path/to/repo"
OUT = sys.argv[2] if len(sys.argv) > 2 else "PA2_output"      # split assignments and result tables go here
SEED = 42
CATEGORIES = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
              "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles",
              "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space",
              "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
              "talk.religion.misc"]

spark = SparkSession.builder.appName("PA2").getOrCreate()

# Label 0-19 follows the alphabetical order of the category folders
labels = spark.createDataFrame(
    [(category, float(i)) for i, category in enumerate(CATEGORIES)],
    "category string, label double")

def feature_stages():
    return [
        RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W", gaps=True, minTokenLength=2, toLowercase=True),
        StopWordsRemover(inputCol="words", outputCol="kept"),
        HashingTF(inputCol="kept", outputCol="counts", numFeatures=4096, binary=False),
        IDF(inputCol="counts", outputCol="features", minDocFreq=2),
    ]

def forest_with(numTrees, maxDepth, minInstancesPerNode):
    return RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=numTrees, maxDepth=maxDepth,
                                  featureSubsetStrategy="sqrt", minInstancesPerNode=minInstancesPerNode,
                                  maxBins=32, seed=SEED)

# Small result tables are written as a single CSV so they can go straight into the report
def save(df, name):
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT}/{name}")

accuracy = MulticlassClassificationEvaluator(metricName="accuracy")
weighted_precision = MulticlassClassificationEvaluator(metricName="weightedPrecision")
weighted_recall = MulticlassClassificationEvaluator(metricName="weightedRecall")
weighted_f1 = MulticlassClassificationEvaluator(metricName="f1")


'''
Task 1: Load, inspect, and partition the data

The archive extracts to one folder per partition, one folder per category, one file per post:
  20news-bydate-train/   11,314 posts   (train + validation)
    alt.atheism/49960
    ...
  20news-bydate-test/     7,532 posts   (final evaluation only)

Each post is an e-mail style header block, a blank line, then the body. Cleaning keeps the Subject line
and the body, drops the other headers (From, Organization, Lines, ...), and collapses whitespace.
------------------------------------------------------
| doc_id | text | label | category | source_split |
------------------------------------------------------
'''
def load(split):
    raw = (spark.read.text(f"{DATA}/20news-bydate-{split}/*/*", wholetext=True)
                .withColumn("path", F.input_file_name())
                .withColumn("value", F.regexp_replace("value", "\r", "")))
    parts = F.split("value", "\n\n", 2)
    return (raw
            .withColumn("category", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 1))
            .withColumn("file_name", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 2))
            .withColumn("doc_id", F.concat_ws("/", F.lit(split), "category", "file_name"))
            .withColumn("source_split", F.lit(split))
            .withColumn("subject", F.regexp_extract(parts.getItem(0), r"(?m)^Subject:[ \t]*(.*)$", 1))
            .withColumn("body", F.coalesce(F.element_at(parts, 2), F.lit("")))
            .withColumn("text", F.trim(F.regexp_replace(F.concat_ws(" ", "subject", "body"), r"\s+", " ")))
            .join(F.broadcast(labels), "category")
            .select("doc_id", "text", "label", "category", "source_split")
            .repartition(8, "doc_id")
            .cache())

official_train = load("train")
official_test = load("test")
official_train.printSchema()
n_train, n_test = official_train.count(), official_test.count()
print("official train:", n_train, "| official test:", n_test)
assert (n_train, n_test) == (11314, 7532), "expected the 11,314 / 7,532 bydate partition"

# Empty documents are kept (they get zero feature vectors) and counted; tokenizing and stop-word removal learn nothing
tokenize = Pipeline(stages=feature_stages()[:2]).fit(official_train)
empties = []
for name, df in [("train", official_train), ("test", official_test)]:
    tokens = tokenize.transform(df)
    empties.append((name, df.filter(F.col("text") == "").count(), tokens.filter(F.size("words") == 0).count(),
                    tokens.filter(F.size("kept") == 0).count()))
empty_counts = spark.createDataFrame(empties, "split string, empty_text long, no_tokens long, no_tokens_after_stop_words long")
empty_counts.show(truncate=False)

# Turns a DataFrame of doc_ids back into full rows, laid out the same way on every machine
def rows_for(ids):
    return official_train.join(ids, "doc_id").repartition(8, "doc_id").sortWithinPartitions("doc_id").cache()

# A fixed row order makes randomSplit give the same split on every run
ids = official_train.select("doc_id").repartition(8, "doc_id").sortWithinPartitions("doc_id")
train_ids, validation_ids = ids.randomSplit([0.8, 0.2], seed=SEED)

train, validation = rows_for(train_ids), rows_for(validation_ids)
n_sub_train, n_validation = train.count(), validation.count()
overlap = train.select("doc_id").intersect(validation.select("doc_id")).count()
train_classes, validation_classes = train.select("label").distinct().count(), validation.select("label").distinct().count()
print("train:", n_sub_train, "| validation:", n_validation, "| overlap:", overlap,
      "| classes in train:", train_classes, "| classes in validation:", validation_classes)
assert overlap == 0 and n_sub_train + n_validation == n_train
assert train_classes == len(CATEGORIES) and validation_classes == len(CATEGORIES)

# randomSplit is not stratified, so look at the class distribution of each piece
class_distribution = (labels
    .join(train.groupBy("label").count().withColumnRenamed("count", "train"), "label", "left")
    .join(validation.groupBy("label").count().withColumnRenamed("count", "validation"), "label", "left")
    .join(official_test.groupBy("label").count().withColumnRenamed("count", "test"), "label", "left")
    .fillna(0)
    .withColumn("train_pct", F.round(100 * F.col("train") / n_sub_train, 2))
    .withColumn("validation_pct", F.round(100 * F.col("validation") / n_validation, 2))
    .select("label", "category", "train", "validation", "test", "train_pct", "validation_pct")
    .orderBy("label"))
class_distribution.show(len(CATEGORIES), truncate=False)
save(class_distribution, "class_distribution")

# Every document's split, for reproducibility
split_assignments = (train.select("doc_id", "label", "category", F.lit("train").alias("split"))
    .unionByName(validation.select("doc_id", "label", "category", F.lit("validation").alias("split")))
    .unionByName(official_test.select("doc_id", "label", "category", F.lit("test").alias("split")))
    .orderBy("doc_id"))
save(split_assignments, "split_assignments")


'''
Task 2: TF-IDF feature extraction

text -> RegexTokenizer -> StopWordsRemover -> HashingTF -> IDF -> features
IDF is fit on the training subset only, then applied to the validation subset.
'''
feature_model = Pipeline(stages=feature_stages()).fit(train)

example = feature_model.transform(train.orderBy("doc_id").limit(1)).first()
print("document:", example["doc_id"], "|", example["category"])
print("excerpt:", example["text"][:300])
print("tokens (%d):" % len(example["words"]), example["words"][:40])
print("after stop-word removal (%d):" % len(example["kept"]), example["kept"][:40])
print("vector size:", example["features"].size, "| nonzero entries:", example["features"].numNonzeros())

train_features = feature_model.transform(train).select("label", "features").cache()
validation_features = feature_model.transform(validation).select("label", "features").cache()
print("cached rows:", train_features.count(), validation_features.count())


'''
Task 3: Baselines

Majority class: the most common category in the training subset, predicted for every validation document.
Initial forest: the feature stages plus a 20-tree, depth-5 forest, fit end to end on the training subset.
'''
majority = train.groupBy("label").count().orderBy(F.desc("count"), "label").first()["label"]
baseline = validation.withColumn("prediction", F.lit(majority))
print("majority category:", CATEGORIES[int(majority)])

start = time.time()
initial_model = Pipeline(stages=feature_stages() + [forest_with(20, 5, 1)]).fit(train)
initial_seconds = time.time() - start
initial_predictions = initial_model.transform(validation).select("label", "prediction").cache()

baselines = spark.createDataFrame([
    ("majority class (%s)" % CATEGORIES[int(majority)], accuracy.evaluate(baseline), weighted_f1.evaluate(baseline), 0.0),
    ("random forest (20 trees, depth 5)", accuracy.evaluate(initial_predictions),
     weighted_f1.evaluate(initial_predictions), initial_seconds),
], "model string, validation_accuracy double, validation_weighted_f1 double, fit_seconds double")
baselines.show(truncate=False)
save(baselines, "baselines")


'''
Task 4: Tuning

30 configurations, each fit on the same training subset and scored on the same validation subset, one at a time.
The feature settings are fixed, so the cached TF-IDF vectors (IDF fit on train only) are reused.
'''
forest = forest_with(20, 5, 1)
grid = (ParamGridBuilder()
        .addGrid(forest.numTrees, [1, 5, 20, 50, 100])
        .addGrid(forest.maxDepth, [5, 10, 25])
        .addGrid(forest.minInstancesPerNode, [1, 5])
        .build())

results = []
for params in grid:
    start = time.time()
    fitted = forest.fit(train_features, params)
    fit_seconds = time.time() - start
    scored = fitted.transform(validation_features).select("label", "prediction").cache()
    results.append((params[forest.numTrees], params[forest.maxDepth], params[forest.minInstancesPerNode],
                    accuracy.evaluate(scored), weighted_f1.evaluate(scored), fit_seconds))
    scored.unpersist()
    print("numTrees %3d  maxDepth %2d  minInstancesPerNode %d  accuracy %.4f  weighted F1 %.4f  fit %.1fs" % results[-1])

tuning = spark.createDataFrame(results, "numTrees int, maxDepth int, minInstancesPerNode int, "
                                        "validation_accuracy double, validation_weighted_f1 double, fit_seconds double")
tuning.orderBy("numTrees", "maxDepth", "minInstancesPerNode").show(len(results), truncate=False)
save(tuning.orderBy("numTrees", "maxDepth", "minInstancesPerNode"), "tuning_results")

# Highest weighted F1 to four decimals; ties go to lower maxDepth, then fewer trees, then larger minInstancesPerNode
best = min(results, key=lambda r: (-round(r[4], 4), r[1], r[0], -r[2]))
best_trees, best_depth, best_min_instances = best[:3]
print("selected: numTrees", best_trees, "| maxDepth", best_depth, "| minInstancesPerNode", best_min_instances,
      "| validation weighted F1 %.4f" % best[4])


'''
Task 5: The final model

Rebuild the whole pipeline with the selected settings and fit it on the entire official training set
(train + validation), which also refits IDF on that set. The official test set is used exactly once, here.
'''
start = time.time()
final_model = Pipeline(stages=feature_stages() + [forest_with(best_trees, best_depth, best_min_instances)]).fit(official_train)
final_seconds = time.time() - start

test_predictions = (final_model.transform(official_test)
                    .select("doc_id", "category", "label", "prediction")
                    .cache())

# Per class: TP = diagonal, support = row sum (TP + FN), predicted = column sum (TP + FP); 0 when a denominator is 0
counts = test_predictions.groupBy("label", "prediction").count().cache()
per_class = (labels
    .join(counts.filter(F.col("label") == F.col("prediction")).select("label", F.col("count").alias("tp")), "label", "left")
    .join(counts.groupBy("label").agg(F.sum("count").alias("support")), "label", "left")
    .join(counts.groupBy("prediction").agg(F.sum("count").alias("predicted")).withColumnRenamed("prediction", "label"), "label", "left")
    .fillna(0, ["tp", "support", "predicted"])
    .withColumn("precision", F.when(F.col("predicted") > 0, F.col("tp") / F.col("predicted")).otherwise(0.0))
    .withColumn("recall", F.when(F.col("support") > 0, F.col("tp") / F.col("support")).otherwise(0.0))
    .withColumn("f1", F.when(F.col("precision") + F.col("recall") > 0,
                             2 * F.col("precision") * F.col("recall") / (F.col("precision") + F.col("recall"))).otherwise(0.0))
    .select("label", "category", "support", "predicted", "tp", "precision", "recall", "f1")
    .orderBy("label")
    .cache())
macro_f1 = per_class.agg(F.avg("f1")).first()[0]

final_metrics = spark.createDataFrame([
    ("numTrees=%d, maxDepth=%d, minInstancesPerNode=%d" % (best_trees, best_depth, best_min_instances),
     accuracy.evaluate(test_predictions), weighted_precision.evaluate(test_predictions),
     weighted_recall.evaluate(test_predictions), weighted_f1.evaluate(test_predictions), macro_f1, final_seconds),
], "configuration string, accuracy double, weighted_precision double, weighted_recall double, "
   "weighted_f1 double, macro_f1 double, fit_seconds double")
final_metrics.show(truncate=False)
save(final_metrics, "final_metrics")

per_class.select("label", "category", "support", F.round("precision", 4).alias("precision"),
                 F.round("recall", 4).alias("recall"), F.round("f1", 4).alias("f1")).show(len(CATEGORIES), truncate=False)
save(per_class, "per_class_metrics")

# Gives each numeric prediction its category name
names = spark.createDataFrame(
    [(float(i), category) for i, category in enumerate(CATEGORIES)],
    "prediction double, predicted string")

# Confusion matrix: rows = true category, columns = predicted category (all 20, even ones never predicted)
confusion = (counts.join(names, "prediction")
                   .join(labels, "label")
                   .groupBy("label", "category").pivot("predicted", CATEGORIES).agg(F.sum("count"))
                   .join(labels, ["label", "category"], "right")
                   .fillna(0)
                   .orderBy("label"))
confusion.show(len(CATEGORIES), truncate=False)
save(confusion, "confusion_matrix")

spark.stop()
