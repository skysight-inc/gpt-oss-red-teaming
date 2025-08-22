import sutro as so
import polars as pl
import json

df = so.get_job_results('job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx')

# convert inference_result to json
df = df.with_columns(
    pl.col("inference_result").str.json_decode().alias("inference_result")
).with_columns(
    pl.col("inference_result").struct.field("content").alias("prompt")
)

# run inference
results = so.infer(
    df,
    model="gpt-oss-20b-no-thinking",
    column="prompt",
    job_priority=1
)


# get results
results = so.get_job_results('job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx', include_inputs=True)
print(results)
for i in results.iter_rows(named=True):
    print("input: ", i['inputs'])
    print("output: ", json.loads(i['inference_result'])['content'])
    print("reasoning: ", json.loads(i['inference_result'])['reasoning_content'])
    print("--------------------------------")