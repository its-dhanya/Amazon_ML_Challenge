"""Run the lexical branch with the vendored Sparkly framework.

For each country, S2+S3 is Sparkly's indexed corpus and S1 is the query table,
so every Source-1 record receives its own lexical top-k list.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

from sparkly.index import LuceneIndex
from sparkly.index_config import IndexConfig
from sparkly.search import Searcher

CLEAN = Path("data/cleaned")
OUT = Path("data/branch_candidates")
INDEX_ROOT = Path("data/sparkly_indexes")
TOP_K = 50


def numbered(df):
    return df.withColumn("_id", F.row_number().over(Window.orderBy("entity_id")).cast("long"))


def run_split(spark: SparkSession, split: str) -> None:
    s1 = spark.read.parquet(str(CLEAN / f"{split}_s1.parquet"))
    target = spark.read.parquet(str(CLEAN / f"{split}_s2.parquet")).unionByName(
        spark.read.parquet(str(CLEAN / f"{split}_s3.parquet"))
    )
    outputs = []
    for country in (r.country for r in s1.select("country").distinct().collect()):
        query = numbered(s1.filter(F.col("country") == country).select("entity_id", "name_clean", "addr_clean"))
        corpus = numbered(target.filter(F.col("country") == country).select("entity_id", "name_clean", "addr_clean"))
        if not query.take(1) or not corpus.take(1):
            continue
        location = INDEX_ROOT / split / str(country)
        shutil.rmtree(location, ignore_errors=True)
        config = IndexConfig(id_col="_id")
        config.add_concat_field("record", ["name_clean", "addr_clean"], ["3gram"])
        lucene = LuceneIndex(str(location), config)
        lucene.upsert_docs(corpus)
        hits = Searcher(lucene).search(query, lucene.get_full_query_spec(), id_col="_id", limit=TOP_K)
        corpus_ids = corpus.select(F.col("_id").alias("corpus_num"), F.col("entity_id").alias("entity_id_s2s3"))
        query_ids = query.select(F.col("_id").alias("query_num"), F.col("entity_id").alias("entity_id_s1"))
        outputs.append(
            hits.select(F.col("id2").alias("query_num"), F.posexplode("id1_list").alias("rank0", "corpus_num"))
            .join(corpus_ids, "corpus_num").join(query_ids, "query_num")
            .select("entity_id_s1", "entity_id_s2s3", (F.col("rank0") + 1).alias("rank"))
        )
    if not outputs:
        raise RuntimeError(f"Sparkly produced no candidates for {split}")
    result = outputs[0]
    for frame in outputs[1:]:
        result = result.unionByName(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    result.dropDuplicates(["entity_id_s1", "entity_id_s2s3"]).write.mode("overwrite").parquet(str(OUT / f"{split}_sparkly.parquet"))


def main() -> None:
    spark = SparkSession.builder.appName("Amazon-ER-Sparkly").getOrCreate()
    for split in ("train", "val", "test"):
        run_split(spark, split)


if __name__ == "__main__":
    main()
