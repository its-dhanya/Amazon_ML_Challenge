"""Splink Fellegi-Sunter scoring over exactly the retrieved candidate union.

Candidate-key duplication is intentional: it turns each retrieved S1--S2/S3
edge into one ``link_only`` comparison.  Splink therefore cannot create pairs
outside ``candidate_pairs_{split}.parquet``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

import splink.comparison_library as cl
from splink import Linker, SettingsCreator, SparkAPI, block_on

CLEAN = Path("data/cleaned")
CAND = Path("data/candidates")
MODEL = Path("data/models/splink_model.json")
OUT = Path("student_resource/output")
THRESHOLD_FILE = Path("data/models/splink_threshold.json")


def settings() -> SettingsCreator:
    return SettingsCreator(
        link_type="link_only",
        unique_id_column_name="candidate_key",
        probability_two_random_records_match=1e-6,
        comparisons=[
            cl.JaroWinklerAtThresholds("name_clean", [0.95, 0.85, 0.72]),
            cl.JaroWinklerAtThresholds("addr_clean", [0.95, 0.85, 0.72]),
            cl.ExactMatch("pin_zip"),
            cl.ExactMatch("state_code"),
            cl.ExactMatch("country"),
        ],
        blocking_rules_to_generate_predictions=[block_on("candidate_key")],
        retain_matching_columns=True,
        retain_intermediate_calculation_columns=True,
    )


def candidate_tables(spark: SparkSession, split: str):
    pairs = spark.read.parquet(str(CAND / f"candidate_pairs_{split}.parquet"))
    s1 = spark.read.parquet(str(CLEAN / f"{split}_s1.parquet"))
    target = spark.read.parquet(str(CLEAN / f"{split}_s2.parquet")).unionByName(
        spark.read.parquet(str(CLEAN / f"{split}_s3.parquet"))
    )
    pairs = pairs.withColumn("candidate_key", F.sha2(F.concat_ws("|", "entity_id_s1", "entity_id_s2s3"), 256))
    cols = ["name_clean", "addr_clean", "pin_zip", "state_code", "country"]
    left = pairs.join(s1.select(F.col("entity_id").alias("entity_id_s1"), *cols), "entity_id_s1").select(
        "candidate_key", F.col("entity_id_s1").alias("entity_id"), *cols
    )
    right = pairs.join(target.select(F.col("entity_id").alias("entity_id_s2s3"), *cols), "entity_id_s2s3").select(
        "candidate_key", F.col("entity_id_s2s3").alias("entity_id"), *cols
    )
    return left, right


def predict(spark: SparkSession, split: str, trained_settings=None):
    left, right = candidate_tables(spark, split)
    linker = Linker([left, right], trained_settings or settings(), db_api=SparkAPI(spark_session=spark), input_table_aliases=["s1", "s2s3"])
    return linker, linker.inference.predict(threshold_match_probability=0.0).as_spark_dataframe()


def choose_threshold(scores, truth, s1_ids) -> float:
    best_t, best_f = 0.5, -1.0
    truth = truth.groupBy("entity_id_s1").agg(F.collect_set("entity_id_s2s3").alias("truth"))
    for threshold in [round(x / 100, 2) for x in range(20, 99)]:
        predicted = scores.filter(F.col("match_probability") >= threshold).groupBy("entity_id_s1").agg(F.collect_set("entity_id_s2s3").alias("pred"))
        rows = s1_ids.join(predicted, "entity_id_s1", "left").join(truth, "entity_id_s1", "left").fillna({"pred": [], "truth": []})
        rows = rows.withColumn("tp", F.size(F.array_intersect("pred", "truth"))).withColumn("p", F.size("pred")).withColumn("r", F.size("truth"))
        precision = F.col("tp") / F.col("p")
        recall = F.col("tp") / F.col("r")
        rows = rows.withColumn(
            "f05",
            F.when((F.col("p") == 0) & (F.col("r") == 0), F.lit(1.0))
            .when(F.col("tp") == 0, F.lit(0.0))
            .otherwise(1.25 * precision * recall / (0.25 * precision + recall)),
        )
        value = rows.agg(F.avg("f05").alias("f")).first().f
        if value > best_f:
            best_t, best_f = threshold, value
    print(f"Best validation macro F0.5={best_f:.4f} at threshold={best_t:.2f}")
    return best_t


def write_submission(scores, threshold: float, spark: SparkSession) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_s1 = spark.read.parquet(str(CLEAN / "test_s1.parquet")).select(F.col("entity_id").alias("source1_entity_id"))
    candidates = scores.groupBy("entity_id_s1").agg(F.concat_ws(",", F.sort_array(F.collect_set("entity_id_s2s3"))).alias("candidate_entity_ids"))
    matches = scores.filter(F.col("match_probability") >= threshold).groupBy("entity_id_s1").agg(F.concat_ws(",", F.sort_array(F.collect_set("entity_id_s2s3"))).alias("matched_entity_ids"))
    def write_one(df, destination: Path) -> None:
        temporary = destination.with_name(destination.stem + "_spark_tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        df.coalesce(1).write.mode("overwrite").option("header", True).option("sep", "\t").csv(str(temporary))
        part = next(temporary.glob("part-*"))
        destination.unlink(missing_ok=True)
        shutil.move(str(part), str(destination))
        shutil.rmtree(temporary)

    write_one(
        test_s1.join(candidates, test_s1.source1_entity_id == candidates.entity_id_s1, "left")
        .select("source1_entity_id", F.coalesce("candidate_entity_ids", F.lit("")).alias("candidate_entity_ids")),
        OUT / "candidate_pairs.tsv",
    )
    write_one(
        test_s1.join(matches, test_s1.source1_entity_id == matches.entity_id_s1, "left")
        .select("source1_entity_id", F.coalesce("matched_entity_ids", F.lit("")).alias("matched_entity_ids")),
        OUT / "matching_results.tsv",
    )


def main() -> None:
    spark = SparkSession.builder.appName("Amazon-ER-Splink").getOrCreate()
    train_linker, _ = predict(spark, "train")
    train_linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)
    train_linker.training.estimate_parameters_using_expectation_maximisation(block_on("candidate_key"))
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    train_linker.misc.save_model_to_json(str(MODEL))
    trained = json.loads(MODEL.read_text())
    _, val_scores = predict(spark, "val", trained)
    val_scores = val_scores.select(F.col("entity_id_l").alias("entity_id_s1"), F.col("entity_id_r").alias("entity_id_s2s3"), "match_probability")
    truth = spark.read.parquet(str(Path("data/processed") / "val_positive_pairs.parquet"))
    s1_ids = spark.read.parquet(str(CLEAN / "val_s1.parquet")).select(F.col("entity_id").alias("entity_id_s1"))
    threshold = choose_threshold(val_scores, truth, s1_ids)
    THRESHOLD_FILE.write_text(json.dumps({"threshold": threshold}))
    _, test_scores = predict(spark, "test", trained)
    test_scores = test_scores.select(F.col("entity_id_l").alias("entity_id_s1"), F.col("entity_id_r").alias("entity_id_s2s3"), "match_probability")
    write_submission(test_scores, threshold, spark)


if __name__ == "__main__":
    main()
