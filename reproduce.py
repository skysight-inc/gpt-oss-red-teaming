import sutro as so
import polars as pl
import json

df = pl.read_parquet("gpt-oss-red-teaming.parquet")
# add a row number column
df = df.with_columns(pl.arange(0, len(df)).alias("row_number"))

rows_to_reproduce = [299609, 279551, 297375, 280741, 279029, 276245, 273952, 294883, 274386, 299609, 297110, 299297, 298615, 233958, 293719, 265228, 234638, 228186, 216308, 204085]

# filter by rows where row_number is in rows_to_reproduce
df = df.filter(pl.col("row_number").is_in(rows_to_reproduce))

results = so.infer(
    df,
    column="attack_prompt",
    model="gpt-oss-20b-no-thinking",
    job_priority=0
)

# print(results)
results = so.get_job_results('job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx', include_inputs=True)

for row in results.iter_rows(named=True):
    print(row["inputs"])
    print("--------------------------------")
    print(json.loads(row["inference_result"])["content"])
    print("--------------------------------")
    print("--------------------------------")