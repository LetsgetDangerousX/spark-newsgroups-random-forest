# PA2: every block I ran in the pyspark shell, in order.



DATA = "file:///s/chopin/n/under/C837364858/cs435/PA2/spark-newsgroups-random-forest"
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IDF, HashingTF, RegexTokenizer, StopWordsRemover
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.sql import SparkSession; spark = SparkSession.builder.appName("PA2").getOrCreate()

CATEGORIES = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
              "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles",
              "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space",
              "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
              "talk.religion.misc"]


# the label table
labels = spark.createDataFrame(
    [(category, float(i)) for i, category in enumerate(CATEGORIES)],
    "category string, label double")
labels.show(20, truncate=False)

# read the raw training files
raw = spark.read.text(f"{DATA}/20news-bydate-train/*/*", wholetext=True).withColumn("path", F.input_file_name())
raw.count()
raw.show(3)

# pull the category and file name out of the path
step = (raw
        .withColumn("category", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 1))
        .withColumn("file_name", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 2)))
step.select("path", "category", "file_name").show(3, truncate=False)

# build the Table 3 columns
train_docs = (step
        .withColumn("doc_id", F.concat_ws("/", F.lit("train"), "category", "file_name"))
        .withColumn("source_split", F.lit("train"))
        .join(F.broadcast(labels), "category")
        .select("doc_id", F.col("value").alias("text"), "label", "category", "source_split"))
train_docs.printSchema()
train_docs.show(3)
train_docs.count()

# Steps 2-4 as one function, for train and test
def load(split):
    raw = spark.read.text(f"{DATA}/20news-bydate-{split}/*/*", wholetext=True).withColumn("path", F.input_file_name())
    return (raw
            .withColumn("category", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 1))
            .withColumn("file_name", F.regexp_extract("path", r"([^/]+)/([^/]+)$", 2))
            .withColumn("doc_id", F.concat_ws("/", F.lit(split), "category", "file_name"))
            .withColumn("source_split", F.lit(split))
            .join(F.broadcast(labels), "category")
            .select("doc_id", F.col("value").alias("text"), "label", "category", "source_split")
            .cache())

official_train = load("train")
official_test = load("test")
print("official train:", official_train.count(), "| official test:", official_test.count())

#  the four feature stages (Task 2 settings)
def feature_stages():
    return [
        RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W", gaps=True, minTokenLength=2, toLowercase=True),
        StopWordsRemover(inputCol="words", outputCol="kept"),
        HashingTF(inputCol="kept", outputCol="counts", numFeatures=4096, binary=False),
        IDF(inputCol="counts", outputCol="features", minDocFreq=2),
    ]

# count empty documents
tokenize = Pipeline(stages=feature_stages()[:2]).fit(official_train)
for name, df in [("train", official_train), ("test", official_test)]:
    tokens = tokenize.transform(df)
    print(name, "| empty text:", df.filter(F.trim("text") == "").count(),
          "| no tokens:", tokens.filter(F.size("words") == 0).count(),
          "| no tokens after stop-word removal:", tokens.filter(F.size("kept") == 0).count())

# split the official train set 80/20
def rows_for(ids):
    return official_train.join(ids, "doc_id").repartition(8, "doc_id").sortWithinPartitions("doc_id").cache()

ids = official_train.select("doc_id").repartition(8, "doc_id").sortWithinPartitions("doc_id")
train_ids, validation_ids = ids.randomSplit([0.8, 0.2], seed=42)

train, validation = rows_for(train_ids), rows_for(validation_ids)
print("train:", train.count(), "| validation:", validation.count())

# Step 9: overlap, class, and distribution checks
print("train/validation overlap:", train.select("doc_id").intersect(validation.select("doc_id")).count())
print("classes in train:", train.select("label").distinct().count(),
      "| classes in validation:", validation.select("label").distinct().count())

(train.groupBy("label", "category").count().withColumnRenamed("count", "train")
      .join(validation.groupBy("label", "category").count().withColumnRenamed("count", "validation"), ["label", "category"], "full_outer")
      .fillna(0)
      .orderBy("label")
      .show(20, truncate=False))

# save the split assignments
(train.select("doc_id", F.lit("train").alias("split"))
      .unionByName(validation.select("doc_id", F.lit("validation").alias("split")))
      .write.mode("overwrite").option("header", True).csv(f"{DATA}/PA2_output/split_assignments"))


# fit TF-IDF on the training subset, inspect one document
feature_model = Pipeline(stages=feature_stages()).fit(train)

row = feature_model.transform(train.orderBy("doc_id").limit(1)).first()
print("document:", row["doc_id"])
print("excerpt:", row["text"][:300])
print("tokens:", row["words"][:40])
print("after stop-word removal:", row["kept"][:40])
print("vector size:", row["features"].size, "| nonzero entries:", row["features"].numNonzeros())


#majority-class baseline
accuracy = MulticlassClassificationEvaluator(metricName="accuracy")
weighted_f1 = MulticlassClassificationEvaluator(metricName="f1")

majority = train.groupBy("label").count().orderBy(F.desc("count"), "label").first()["label"]
baseline = validation.withColumn("prediction", F.lit(majority))
print("majority category:", CATEGORIES[int(majority)], "| accuracy:", round(accuracy.evaluate(baseline), 4),
      "| weighted F1:", round(weighted_f1.evaluate(baseline), 4))

# initial Random Forest
forest = RandomForestClassifier(
    labelCol="label", featuresCol="features",
    numTrees=20, maxDepth=5, featureSubsetStrategy="sqrt",
    minInstancesPerNode=1, maxBins=32, seed=42)

start = time.time()
model = Pipeline(stages=feature_stages() + [forest]).fit(train)
fit_seconds = time.time() - start

predictions = model.transform(validation).select("label", "prediction").cache()
print("forest accuracy:", round(accuracy.evaluate(predictions), 4),
      "| weighted F1:", round(weighted_f1.evaluate(predictions), 4), "| fit %.1fs" % fit_seconds)


#  grid + cached TF-IDF vectors
grid = (ParamGridBuilder()
        .addGrid(forest.numTrees, [1, 5, 20, 50, 100])
        .addGrid(forest.maxDepth, [5, 10, 25])
        .addGrid(forest.minInstancesPerNode, [1, 5])
        .build())
print("configurations:", len(grid))

train_features = feature_model.transform(train).select("label", "features").cache()
validation_features = feature_model.transform(validation).select("label", "features").cache()
print("cached rows:", train_features.count(), validation_features.count())

#  the tuning loop
results = []
for params in grid:
    start = time.time()
    fitted = forest.fit(train_features, params)
    fit_seconds = time.time() - start
    scored = fitted.transform(validation_features).select("label", "prediction").cache()
    r = {"numTrees": params[forest.numTrees],
         "maxDepth": params[forest.maxDepth],
         "minInstancesPerNode": params[forest.minInstancesPerNode],
         "accuracy": accuracy.evaluate(scored),
         "weighted_f1": weighted_f1.evaluate(scored),
         "fit_seconds": fit_seconds}
    results.append(r)
    print("numTrees %3d  maxDepth %2d  minInstancesPerNode %d  accuracy %.4f  weighted F1 %.4f  fit %.1fs"
          % (r["numTrees"], r["maxDepth"], r["minInstancesPerNode"], r["accuracy"], r["weighted_f1"], r["fit_seconds"]))

#  pick the configuration (highest weighted F1; ties: lower depth, fewer trees, larger minInstancesPerNode)
best = min(results, key=lambda r: (-round(r["weighted_f1"], 4), r["maxDepth"], r["numTrees"], -r["minInstancesPerNode"]))
print("selected: numTrees", best["numTrees"], "| maxDepth", best["maxDepth"],
      "| minInstancesPerNode", best["minInstancesPerNode"], "| weighted F1 %.4f" % best["weighted_f1"])


#refit on the full official train set, score the test set
final_forest = RandomForestClassifier(labelCol="label", featuresCol="features",
                                      numTrees=best["numTrees"], maxDepth=best["maxDepth"],
                                      minInstancesPerNode=best["minInstancesPerNode"],
                                      featureSubsetStrategy="sqrt", maxBins=32, seed=42)
final_model = Pipeline(stages=feature_stages() + [final_forest]).fit(official_train)

test_predictions = final_model.transform(official_test).select("doc_id", "category", "label", "prediction").cache()
print("test accuracy:", round(accuracy.evaluate(test_predictions), 4))
print("weighted precision:", round(MulticlassClassificationEvaluator(metricName="weightedPrecision").evaluate(test_predictions), 4))
print("weighted recall:", round(MulticlassClassificationEvaluator(metricName="weightedRecall").evaluate(test_predictions), 4))
print("weighted F1:", round(weighted_f1.evaluate(test_predictions), 4))

#20x20 confusion matrix (rows = true, columns = predicted)
names = spark.createDataFrame([(float(i), c) for i, c in enumerate(CATEGORIES)], "prediction double, predicted string")

(test_predictions.join(names, "prediction")
                 .groupBy("label", "category").pivot("predicted", CATEGORIES).count().fillna(0)
                 .orderBy("label")
                 .show(20, truncate=False))

#per-class precision, recall, F1, and macro F1
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
    .orderBy("label"))
per_class.select("label", "category", F.round("precision", 4).alias("precision"),
                 F.round("recall", 4).alias("recall"), F.round("f1", 4).alias("f1")).show(20, truncate=False)
print("macro F1:", round(per_class.agg(F.avg("f1")).first()[0], 4))
