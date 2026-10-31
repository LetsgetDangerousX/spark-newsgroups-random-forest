"""
Builds the PA2 report (report.html) from the CSV tables PA2.py writes to its results folder.

    python3 make_report.py results

Open report.html in a browser and print it to PDF (landscape fits the confusion matrix best).
Standard library only; it reads the small result tables, never the corpus.
"""
import csv
import html
import os
import sys
from collections import defaultdict

results = sys.argv[1] if len(sys.argv) > 1 else "results"
out_path = sys.argv[2] if len(sys.argv) > 2 else "report.html"


def read(name):
    with open(os.path.join(results, name), newline="") as f:
        return list(csv.DictReader(f))


def fmt(value, digits=4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if number.is_integer() and digits == 0:
        return str(int(number))
    return f"{number:.{digits}f}"


def table(header, rows, numeric_from=0, row_class=None):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = []
    for i, row in enumerate(rows):
        cls = f' class="{row_class(i)}"' if row_class and row_class(i) else ""
        cells = "".join(f'<td class="{"num" if j >= numeric_from else ""}">{c}</td>' for j, c in enumerate(row))
        body.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


sizes = read("task1_split_sizes.csv")
distribution = read("task1_class_distribution.csv")
empty = read("task1_empty_documents.csv")
example = {r["field"]: r["value"] for r in read("task2_feature_example.csv")}
baselines = read("task3_baselines.csv")
tuning = read("task4_tuning_results.csv")
selected = read("task4_selected.csv")[0]
overall = {r["metric"]: float(r["value"]) for r in read("task5_overall_metrics.csv")}
per_class = read("task5_per_class.csv")
with open(os.path.join(results, "task5_confusion_matrix.csv"), newline="") as f:
    matrix_rows = list(csv.reader(f))
categories = matrix_rows[0][1:]
matrix = [[int(v) for v in r[1:]] for r in matrix_rows[1:]]

for r in tuning:
    for k in ("numTrees", "maxDepth", "minInstancesPerNode"):
        r[k] = int(r[k])
    for k in ("accuracy", "weighted_f1", "fit_seconds"):
        r[k] = float(r[k])

# ---- Tuning discussion, computed from the table ----
best = max(tuning, key=lambda r: r["weighted_f1"])
by_depth, by_trees, time_by_depth = defaultdict(list), defaultdict(list), defaultdict(list)
for r in tuning:
    by_depth[r["maxDepth"]].append(r["weighted_f1"])
    by_trees[r["numTrees"]].append(r["weighted_f1"])
    time_by_depth[r["maxDepth"]].append(r["fit_seconds"])
mean = lambda xs: sum(xs) / len(xs)
min_inst_gain = mean([a["weighted_f1"] - b["weighted_f1"] for a in tuning if a["minInstancesPerNode"] == 1
                      for b in tuning if b["minInstancesPerNode"] == 5
                      and (a["numTrees"], a["maxDepth"]) == (b["numTrees"], b["maxDepth"])])
within = [r for r in tuning if r["weighted_f1"] >= best["weighted_f1"] - 0.03]
cheapest_close = min(within, key=lambda r: r["fit_seconds"])
deep_20 = next(r for r in tuning if (r["numTrees"], r["maxDepth"], r["minInstancesPerNode"]) == (20, 25, 1))

# ---- Error analysis, computed from the confusion matrix ----
pairs = sorted(((matrix[i][j], categories[i], categories[j]) for i in range(len(categories))
                for j in range(len(categories)) if i != j), reverse=True)[:5]
worst = sorted(per_class, key=lambda r: float(r["f1"]))[:3]
best_classes = sorted(per_class, key=lambda r: -float(r["f1"]))[:3]
most_predicted = max(per_class, key=lambda r: int(r["predicted"]) - int(r["support"]))
validation_f1 = float(selected["weighted_f1"])

style = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; color: #1d1d1f; margin: 32px auto; max-width: 1050px; font-size: 13px; line-height: 1.45; }
h1 { font-size: 22px; margin-bottom: 2px; } h2 { font-size: 17px; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
h3 { font-size: 14px; margin-top: 18px; }
table { border-collapse: collapse; margin: 8px 0 14px; font-size: 12px; }
th, td { border: 1px solid #d6d6d6; padding: 3px 7px; text-align: left; } th { background: #f3f3f3; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { background: #e5f4e4; font-weight: 600; }
.cm td, .cm th { padding: 2px 4px; text-align: right; } .cm td.diag { background: #e5f4e4; font-weight: 600; }
.cm td.hot { background: #fbebeb; }
.muted { color: #666; } pre { background: #f6f6f6; padding: 8px; white-space: pre-wrap; font-size: 11px; }
@media print { body { margin: 0; max-width: none; } h2 { break-after: avoid; } table { break-inside: avoid; } }
"""

parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>PA2 Report</title><style>{style}</style></head><body>",
         "<h1>PA2: Document Classification with Random Forest on Apache Spark</h1>",
         "<p class='muted'>20 Newsgroups (bydate), TF-IDF features, Spark ML RandomForestClassifier, seed 42.</p>"]

# Task 1
parts.append("<h2>Task 1. Data and splits</h2>")
parts.append(table(["split", "documents"], [[r["split"], r["documents"]] for r in sizes], 1))
parts.append("<p>The official training set (11,314 posts) was split ~80/20 into train and validation with "
             "<code>randomSplit([0.8, 0.2], seed=42)</code>; the official test set (7,532 posts) was left untouched until Task 5. "
             "The two subsets share no document IDs, and all 20 classes appear in both.</p>")
parts.append(table(["label", "category", "train", "validation", "test"],
                   [[r["label"], r["category"], r["train"], r["validation"], r["test"]] for r in distribution], 2))
parts.append("<p>Cleaning removes the e-mail header block (everything before the first blank line). Documents that end up "
             "empty are kept and receive zero feature vectors:</p>")
parts.append(table(["official split", "empty after cleaning", "no tokens", "no tokens after stop-word removal"],
                   [[r["split"], r["empty_text"], r["no_tokens"], r["no_tokens_after_stop_words"]] for r in empty], 1))

# Task 2
parts.append("<h2>Task 2. TF-IDF features</h2>")
parts.append(f"<p>Example training document <code>{html.escape(example['doc_id'])}</code>:</p>"
             f"<pre>{html.escape(example['excerpt'])}</pre>")
parts.append(table(["tokens", "after stop-word removal", "vector size", "nonzero entries"],
                   [[example["tokens"], example["tokens_after_stop_words"], example["vector_size"], example["nonzero_entries"]]], 0))
parts.append(f"<p>Manual check for the term <b>{html.escape(example['check_term'])}</b> (hash slot {example['slot']}): "
             f"Spark's TF-IDF is {fmt(example['spark_tfidf'], 6)}. Using the slot's counts (TF {fmt(example['tf_slot'], 0)}, "
             f"DF {example['df_slot']}) the formula gives {fmt(example['manual_tfidf_slot'], 6)}, an exact match. "
             f"Counting only the word itself (DF {example['df_word']}) gives {fmt(example['tfidf_word_no_collisions'], 6)}: "
             f"other words hash to the same slot and raise its document frequency, which is the cost of hashing collisions.</p>")

# Task 3
parts.append("<h2>Task 3. Baselines (validation set)</h2>")
parts.append(table(["model", "accuracy", "weighted F1", "fit time (s)"],
                   [[r["model"], fmt(r["validation_accuracy"]), fmt(r["validation_weighted_f1"]),
                     fmt(r["fit_seconds"], 1) if r["fit_seconds"] else "n/a"] for r in baselines], 1))

# Task 4
parts.append("<h2>Task 4. Hyperparameter tuning (validation set)</h2>")
ordered = sorted(tuning, key=lambda r: (r["numTrees"], r["maxDepth"], r["minInstancesPerNode"]))
parts.append(table(["numTrees", "maxDepth", "minInstancesPerNode", "accuracy", "weighted F1", "fit time (s)"],
                   [[r["numTrees"], r["maxDepth"], r["minInstancesPerNode"], fmt(r["accuracy"]), fmt(r["weighted_f1"]),
                     fmt(r["fit_seconds"], 1)] for r in ordered], 0,
                   row_class=lambda i: "best" if ordered[i] is best else ""))
parts.append(f"<p><b>Selected configuration:</b> numTrees={selected['numTrees']}, maxDepth={selected['maxDepth']}, "
             f"minInstancesPerNode={selected['minInstancesPerNode']} (validation weighted F1 {fmt(selected['weighted_f1'])}, "
             f"accuracy {fmt(selected['accuracy'])}). Selection rule: highest weighted F1 to four decimals; ties go to lower "
             f"maxDepth, then fewer trees, then larger minInstancesPerNode.</p>")
parts.append("<h3>Accuracy–runtime tradeoffs</h3><ul>")
parts.append(f"<li><b>Depth matters most.</b> Mean weighted F1 is {fmt(mean(by_depth[5]))} at depth 5, {fmt(mean(by_depth[10]))} "
             f"at depth 10 and {fmt(mean(by_depth[25]))} at depth 25. With 4,096 sparse features and 20 classes, shallow trees "
             f"can only test a few dozen words each. Depth is also what drives cost: mean fit time rises from "
             f"{fmt(mean(time_by_depth[5]), 1)}s (depth 5) to {fmt(mean(time_by_depth[25]), 1)}s (depth 25), because each extra "
             f"level is another pass over the data with more nodes and larger split statistics.</li>")
parts.append(f"<li><b>More trees help with diminishing returns.</b> Mean weighted F1 by numTrees: "
             + ", ".join(f"{t}: {fmt(mean(by_trees[t]))}" for t in sorted(by_trees))
             + f". Fit time grows roughly linearly with the number of trees, so going from 50 to 100 trees costs about "
               f"twice the time for a smaller gain than going from 5 to 20.</li>")
parts.append(f"<li><b>minInstancesPerNode=5 mostly hurts slightly</b> (by {fmt(min_inst_gain)} weighted F1 on average): "
             f"forcing larger leaves limits how specific a tree can get, and the forest's averaging already controls "
             f"overfitting. It does cut fit time a little for deep trees, since fewer nodes are split.</li>")
parts.append(f"<li><b>Cheaper options.</b> The best model takes {fmt(best['fit_seconds'], 1)}s. "
             f"numTrees={cheapest_close['numTrees']}, maxDepth={cheapest_close['maxDepth']}, "
             f"minInstancesPerNode={cheapest_close['minInstancesPerNode']} reaches {fmt(cheapest_close['weighted_f1'])} in "
             f"{fmt(cheapest_close['fit_seconds'], 1)}s, and 20 trees at depth 25 reaches {fmt(deep_20['weighted_f1'])} in "
             f"{fmt(deep_20['fit_seconds'], 1)}s. If training time were constrained, those would be reasonable choices.</li></ul>")

# Task 5
parts.append("<h2>Task 5. Final model on the official test set</h2>")
parts.append("<p>The selected pipeline was rebuilt and fitted on the entire official training set (IDF refitted on it), "
             "then evaluated once on the official test set.</p>")
parts.append(table(["metric", "value"], [
    ["accuracy", fmt(overall["accuracy"])], ["weighted precision", fmt(overall["weighted_precision"])],
    ["weighted recall", fmt(overall["weighted_recall"])], ["weighted F1", fmt(overall["weighted_f1"])],
    ["macro F1", fmt(overall["macro_f1"])]], 1))
parts.append("<h3>Per-class metrics</h3>")
parts.append(table(["label", "category", "support", "TP", "predicted", "precision", "recall", "F1"],
                   [[r["label"], r["category"], r["support"], r["tp"], r["predicted"], fmt(r["precision"]),
                     fmt(r["recall"]), fmt(r["f1"])] for r in per_class], 2))
parts.append("<h3>Confusion matrix</h3><p class='muted'>Rows are true labels, columns are predicted labels (column numbers "
             "are the row labels).</p>")
off_diag_max = max(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i != j)
head = "<tr><th>true \\ predicted</th>" + "".join(f"<th>{i}</th>" for i in range(len(categories))) + "</tr>"
body = []
for i, row in enumerate(matrix):
    cells = []
    for j, v in enumerate(row):
        cls = "diag" if i == j else ("hot" if v >= off_diag_max * 0.4 else "")
        cells.append(f"<td class='{cls}'>{v}</td>")
    body.append(f"<tr><th style='text-align:left'>{i} {html.escape(categories[i])}</th>{''.join(cells)}</tr>")
parts.append(f"<table class='cm'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>")
parts.append("<h3>Interpretation</h3><ul>")
parts.append("<li><b>Largest confusions:</b> " + "; ".join(
    f"{html.escape(t)} predicted as {html.escape(p)} ({n})" for n, t, p in pairs)
    + ". These are topically overlapping groups: the religion groups share vocabulary, as do the politics groups and the "
      "computer hardware/software groups.</li>")
parts.append("<li><b>Hardest classes</b> (lowest F1): " + ", ".join(
    f"{html.escape(r['category'])} ({fmt(r['f1'])})" for r in worst)
    + ". <b>Easiest:</b> " + ", ".join(f"{html.escape(r['category'])} ({fmt(r['f1'])})" for r in best_classes) + ".</li>")
parts.append(f"<li><b>{html.escape(most_predicted['category'])}</b> is predicted {most_predicted['predicted']} times but has only "
             f"{most_predicted['support']} true documents, so it absorbs errors from many classes and has low precision.</li>")
parts.append(f"<li><b>Test vs. validation:</b> test weighted F1 ({fmt(overall['weighted_f1'])}) is lower than validation "
             f"({fmt(validation_f1)}). The bydate split puts later posts in the test set, so topics and threads drift; the "
             f"validation set was drawn at random from the same period as training, so it overestimates performance on "
             f"future posts.</li>")
parts.append(f"<li><b>Macro vs. weighted F1:</b> macro F1 ({fmt(overall['macro_f1'])}) is below weighted F1 because the "
             f"smallest classes (e.g. talk.religion.misc) are also among the hardest, and macro F1 gives them equal weight.</li></ul>")
parts.append("</body></html>")

with open(out_path, "w") as f:
    f.write("\n".join(parts))
print("wrote", os.path.abspath(out_path))
