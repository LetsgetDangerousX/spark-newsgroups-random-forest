import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IDF, HashingTF, RegexTokenizer, StopWordsRemover
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA = "/user/C837364858/PA2"      # local mode: "file:///full/path/to/PA2"
OUT = "PA2_output"                 # where the split assignments are saved
CATEGORIES = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
              "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles",
              "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space",
              "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
              "talk.religion.misc"]

spark = SparkSession.builder.appName("PA2").getOrCreate()

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

'''
Task 1: Load, inspect, and partition the data

Our data is structured as such:
--PA2
  |
  --20news-bydate-train
    --alt.atheism
      *49960*
    ...
  --20news-bydate-test
    ...

We read both official partitions into DataFrames of the form:
------------------------------------------------------
| doc_id | text | label | category | source_split |
------------------------------------------------------
'''

def load(split):
    raw = spark.read.text(f"{DATA}/20news-bydate-{split}/*/*", wholetext=True).withColumn("path", F.input_file_name())
    return (raw
            .withColumn("category", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 1))
            .withColumn("file_name", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 2))
            .withColumn("doc_id", F.concat_ws("/", F.lit(split), "category", "file_name"))
            .withColumn("source_split", F.lit(split))
            .join(F.broadcast(labels), "category")
            .select("doc_id", F.col("value").alias("text"), "label", "category", "source_split")
            .repartition(8, "doc_id")
            .cache())

official_train = load("train")
official_test = load("test")
official_train.printSchema()
print("official train:", official_train.count(), "| official test:", official_test.count())

# Documents that become empty after tokenization are kept and counted
tokenize = Pipeline(stages=feature_stages()[:2]).fit(official_train)
for name, df in [("train", official_train), ("test", official_test)]:
    tokens = tokenize.transform(df)
    print(name, "| empty text:", df.filter(F.trim("text") == "").count(),
          "| no tokens:", tokens.filter(F.size("words") == 0).count(),
          "| no tokens after stop-word removal:", tokens.filter(F.size("kept") == 0).count())

# Turns a DataFrame of doc_ids back into full rows, laid out the same way on every machine
def rows_for(ids):
    return official_train.join(ids, "doc_id").repartition(8, "doc_id").sortWithinPartitions("doc_id").cache()

ids = official_train.select("doc_id").repartition(8, "doc_id").sortWithinPartitions("doc_id")
train_ids, validation_ids = ids.randomSplit([0.8, 0.2], seed=42)

train, validation = rows_for(train_ids), rows_for(validation_ids)
print("train:", train.count(), "| validation:", validation.count())
print("train/validation overlap:", train.select("doc_id").intersect(validation.select("doc_id")).count())

(train.groupBy("label", "category").count().withColumnRenamed("count", "train")
      .join(validation.groupBy("label", "category").count().withColumnRenamed("count", "validation"), ["label", "category"], "full_outer")
      .fillna(0)
      .orderBy("label")
      .show(len(CATEGORIES), truncate=False))

(train.select("doc_id", F.lit("train").alias("split"))
      .unionByName(validation.select("doc_id", F.lit("validation").alias("split")))
      .write.mode("overwrite").option("header", True).csv(f"{OUT}/split_assignments"))

'''
Task 2: TF-IDF feature extraction

IDF is fit on the training subset only.
'''
feature_model = Pipeline(stages=feature_stages()).fit(train)

row = feature_model.transform(train.orderBy("doc_id").limit(1)).first()
print("article:", row["doc_id"])
print("excerpt:", row["text"][:300])
print("tokens:", row["words"])
print("after stop-word removal:", row["kept"])
print("vector size:", row["features"].size, "| nonzero entries:", row["features"].numNonzeros())

'''
Task 3: Baselines
'''
accuracy = MulticlassClassificationEvaluator(metricName="accuracy")
weighted_f1 = MulticlassClassificationEvaluator(metricName="f1")

majority = train.groupBy("label").count().orderBy(F.desc("count"), "label").first()["label"]
baseline = validation.withColumn("prediction", F.lit(majority))
print("majority category:", CATEGORIES[int(majority)], "| accuracy:", round(accuracy.evaluate(baseline), 4),
      "| weighted F1:", round(weighted_f1.evaluate(baseline), 4))

forest = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=20,
    maxDepth=5,
    featureSubsetStrategy="sqrt",
    minInstancesPerNode=1,
    maxBins=32,
    seed=42,
)
start = time.time()
model = Pipeline(stages=feature_stages() + [forest]).fit(train)
fit_seconds = time.time() - start

predictions = model.transform(validation).select("label", "prediction").cache()
print("forest accuracy:", round(accuracy.evaluate(predictions), 4),
      "| weighted F1:", round(weighted_f1.evaluate(predictions), 4), "| fit %.1fs" % fit_seconds)

'''
Task 4: Tuning

The feature settings stay fixed, so the TF-IDF vectors (IDF fit on train only) are cached and reused.
'''
forest = RandomForestClassifier(labelCol="label", featuresCol="features", featureSubsetStrategy="sqrt", maxBins=32, seed=42)
grid = (ParamGridBuilder()
        .addGrid(forest.numTrees, [1, 5, 20, 50, 100])
        .addGrid(forest.maxDepth, [5, 10, 25])
        .addGrid(forest.minInstancesPerNode, [1, 5])
        .build())

train_features = feature_model.transform(train).select("label", "features").cache()
validation_features = feature_model.transform(validation).select("label", "features").cache()
print("cached rows:", train_features.count(), validation_features.count())

results = []
for params in grid:
    start = time.time()
    fitted = forest.fit(train_features, params)
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

for r in results:
    print("numTrees %3d  maxDepth %2d  minInstancesPerNode %d  accuracy %.4f  weighted F1 %.4f  fit %.1fs"
          % (r["numTrees"], r["maxDepth"], r["minInstancesPerNode"], r["accuracy"], r["weighted_f1"], r["fit_seconds"]))

# Highest weighted F1 to four decimals; ties: lower maxDepth, then fewer trees, then larger minInstancesPerNode
best = min(results, key=lambda r: (-round(r["weighted_f1"], 4), r["maxDepth"], r["numTrees"], -r["minInstancesPerNode"]))
print("selected: numTrees", best["numTrees"], "maxDepth", best["maxDepth"], "minInstancesPerNode", best["minInstancesPerNode"])

'''
Task 5: The final model

Rebuild the selected pipeline, fit it on the entire official training set (this refits IDF),
and evaluate on the official test set.
'''
final_forest = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=best["numTrees"],
                                      maxDepth=best["maxDepth"], minInstancesPerNode=best["minInstancesPerNode"],
                                      featureSubsetStrategy="sqrt", maxBins=32, seed=42)
final_model = Pipeline(stages=feature_stages() + [final_forest]).fit(official_train)

test_predictions = (final_model.transform(official_test)
                    .select("doc_id", "category", "label", "prediction")
                    .cache())
print("test accuracy:", round(accuracy.evaluate(test_predictions), 4))
print("weighted precision:", round(MulticlassClassificationEvaluator(metricName="weightedPrecision").evaluate(test_predictions), 4))
print("weighted recall:", round(MulticlassClassificationEvaluator(metricName="weightedRecall").evaluate(test_predictions), 4))
print("weighted F1:", round(weighted_f1.evaluate(test_predictions), 4))

# Gives each numeric prediction its category name
names = spark.createDataFrame(
    [(float(i), category) for i, category in enumerate(CATEGORIES)],
    "prediction double, predicted string")

# Confusion matrix: rows = true category, columns = predicted category
(test_predictions.join(names, "prediction")
                 .groupBy("label", "category").pivot("predicted", CATEGORIES).count().fillna(0)
                 .orderBy("label")
                 .show(len(CATEGORIES), truncate=False))

# Per class: TP = diagonal, support = row sum (TP + FN), predicted = column sum (TP + FP); 0 when a denominator is 0
counts = test_predictions.groupBy("label", "prediction").count()
per_class = (labels
             .join(counts.filter(F.col("label") == F.col("prediction")).select("label", F.col("count").alias("tp")), "label", "left")
             .join(counts.groupBy("label").agg(F.sum("count").alias("support")), "label", "left")
             .join(counts.groupBy("prediction").agg(F.sum("count").alias("predicted")).withColumnRenamed("prediction", "label"), "label", "left")
             .fillna(0, ["tp", "support", "predicted"])
             .withColumn("precision", F.when(F.col("predicted") > 0, F.col("tp") / F.col("predicted")).otherwise(0.0))
             .withColumn("recall", F.when(F.col("support") > 0, F.col("tp") / F.col("support")).otherwise(0.0))
             .withColumn("f1", F.when(F.col("precision") + F.col("recall") > 0,
                                      2 * F.col("precision") * F.col("recall") / (F.col("precision") + F.col("recall"))).otherwise(0.0))
             .orderBy("label")
             .cache())

per_class.select("label", "category", F.round("precision", 4).alias("precision"),
                 F.round("recall", 4).alias("recall"), F.round("f1", 4).alias("f1")).show(len(CATEGORIES), truncate=False)
print("macro F1:", round(per_class.agg(F.avg("f1")).first()[0], 4))

spark.stop()
