import polars as pl
import json

df = pl.read_parquet("gpt-oss-red-teaming.parquet")

rows = [299842, 299297, 15115, 233958, 234638]

# filter
df = df.filter(pl.col("row_number").is_in(rows))

issue_format = {
    "schema_version": "1.0.0",
    "issue_title": "",
    "model": {
        "name": "gpt-oss-20b",
        "version": "2025-08-05",
        "provider": "openai",
        "parameters": {
            "temperature": 1.0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 4096,
            "reasoning_level": "low",
        },
    },
    "environment": {
        "endpoint": "Sutro API (https://sutro.sh/)",
        "hardware": "1x H100 80GB",
        "libraries": "vLLM",
    },
    "issue_summary": {
        "title:": "",
        "topic_area": "",
        "overview": "",
        "desired_behavior": "",
        "observed_behavior": "",
        "self_assessed_severity": "",
        "self_assessed_severity_justification": "",
        "self_assessed_breadth": "",
        "self_assessed_breadth_justification": "",
    },
    "harmony_response_walkthrough": [],
    "steps_to_reproduce": [
        "Set sampling parameters as listed",
        "Add the default OpenAI system prompt with reasoning effort set to LOW to the harmony messages.",
        "Add the prompt as a user message to the harmony messages.",
        "Add an empty string using the analysis channel and assistant role to the harmony messages.",
        "Render the harmony conversation and generate the final assistant message.",
    ],
    "notes": ""
}

count = 0
for row in df.iter_rows(named=True):
    with open(f"findings/sutro.findings.{count}.jsonl", "w") as f:
        issue = issue_format.copy()
        issue["harmony_response_walkthrough"] = [json.loads(row["harmony_conversation"])]
        issue["issue_summary"]["topic_area"] = row["exploit_dimension"]
        json.dump(issue, f, indent=4)
        f.write("\n")
    count += 1

# only keep attack_prompt and attack_reply
df = df.select(["attack_prompt", "attack_reply"])

# write to a csv
df.write_csv("attack_prompt_and_reply.csv")