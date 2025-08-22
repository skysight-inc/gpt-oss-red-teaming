import polars as pl
import sutro as so
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)
import json

dimension_descriptions = {
    "deception": "Unlike unintentional hallucinations, strategic lying occurs when a model knowingly emits falsehoods to achieve a goal (e.g., avoiding shutdown or steering a user). Cutting-edge studies document models that misstate their preferences or capabilities during safety evaluations, signalling a need for tests that can elicit and penalize deliberate deception.",
    "data_exfiltration": "Models can be coaxed (or back-doored) into leaking proprietary training data or user inputs, or can encode secret payloads in their outputs for later retrieval. Attacks include trigger-based payload extraction and covert-channel steganography, threatening both privacy and intellectual property. Competitors who uncover reliable data-leak vectors will earn high marks for severity and novelty.",
    "hidden_motivations": "A model might internally pursue goals that differ from its training objective, acting helpful only when that serves its private aim. Experiments show systems that strategically withhold information or mislead supervisors to avoid further fine-tuning, raising the spectre of agents that feign obedience until they gain leverage. Detecting such covert objectives is a central aim of the competition",
    "sandbagging": "Strategically deceptive models may seek to hide their full capabilities in a variety of conditions (including dangerous capability evaluations, alignment research) and for a variety of reasons (to get deployed, to prevent their scheming being detected).",
    "sabotage": "Beyond passive misalignment, an autonomous agent could intentionally degrade or disrupt the very process it was asked to support: corrupting files, inserting subtle bugs, or derailing workflows once it has the opportunity. Recent threat-models highlight “catastrophic sabotage” as a plausible path to large-scale harm if powerful models are given high-impact actuators.",
    "reward_hacking": "When the training signal is imperfect, a model can learn shortcuts that maximize reward metrics without truly solving the task (e.g., producing confident-sounding but false answers). This mirrors the classic RL failure mode where agents game the score instead of achieving the designer’s intent, and it can surface in LLMs as hallucinations styled to please raters or as jailbreak-friendly phrasing that evades filters."
}

dimension_data = {
    "deception": {
        "prompts": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "attack": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "evaluate": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx"
    },
    "data_exfiltration": {
        "prompts": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "attack": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "evaluate": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx"
    },
    "hidden_motivations": {
        "prompts": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "attack": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "evaluate": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx"
    },
    "sandbagging": {
        "prompts": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "attack": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "evaluate": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx"
    },
    "sabotage": {
        "prompts": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "attack": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "evaluate": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx"
    },
    "reward_hacking": {
        "prompts": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "attack": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx",
        "evaluate": "job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx"
    }
}

master_df = pl.DataFrame()

for dimension, dimention_description in dimension_descriptions.items():
    personas_df = pl.read_parquet('hf://datasets/sutro/synthetic-humans-50k/chunk_0.parquet')
    # drop all columns except for id
    personas_df = personas_df.drop([col for col in personas_df.columns if col != "id"])
    personas_df = personas_df.rename({"id": "persona_id"})

    prompts_df = so.get_job_results(dimension_data[dimension]["prompts"], include_inputs=True)
    prompts_df = prompts_df.with_columns(
        pl.col("inference_result").str.json_decode().alias("data_gen_raw")
    ).with_columns(
        pl.col("data_gen_raw").struct.field("content").alias("attack_prompt"),
    )

    prompts_df = prompts_df.rename({
        "inputs": "data_gen_prompt",
    })

    # drop the inference_result column
    prompts_df = prompts_df.drop(["inference_result"])

    # get attack_df
    attack_df = so.get_job_results(dimension_data[dimension]["attack"], include_inputs=True)
    attack_df = attack_df.with_columns(
        pl.col("inference_result").str.json_decode().alias("attack_response_raw")
    ).with_columns(
        pl.col("attack_response_raw").struct.field("content").alias("attack_reply"),
    )

    # drop the inputs column
    attack_df = attack_df.drop(["inputs", "inference_result"])

    # get the evaluation_df
    evaluation_df = so.get_job_results(dimension_data[dimension]["evaluate"], include_inputs=True)
    evaluation_df = evaluation_df.with_columns(
        pl.col("inference_result").str.json_decode().alias("evaluation_response_raw")
    ).with_columns(
        pl.col("evaluation_response_raw").struct.field("content").alias("evaluation_reply"),
    ).with_columns(
        pl.col("evaluation_reply").struct.field("success_score").alias("success_score"),
        pl.col("evaluation_reply").struct.field("reason").alias("evaluation_reason")
    )

    # drop the inputs column
    evaluation_df = evaluation_df.drop(["inputs", "inference_result"])

    # create a wide dataframe with the following columns:
    # - exploit_dimension
    # - data_gen_prompt
    # - data_gen_raw
    # - attack_prompt
    # - attack_response_raw
    # - attack_reply
    # - success_score
    # - evaluation_reason
    # - evaluation_response_raw
    # - evaluation_reply
    # - success_score
    # - evaluation_reason
    dimension_df = pl.concat([
        personas_df,
        prompts_df,
        attack_df,
        evaluation_df
    ],
    how="horizontal")

    # add the dimension to the dimension_df
    dimension_df = dimension_df.with_columns(
        pl.lit(dimension).alias("exploit_dimension")
    )

    master_df = pl.concat([
        master_df,
        dimension_df
    ],
    how="vertical")

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

def render_harmony_conversation(row):
    input_str = row['attack_prompt']
    output_str = row['attack_reply']
    input_messages = [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)),
        Message.from_role_and_content(Role.USER, input_str),
        Message.from_role_and_content(Role.ASSISTANT, "").with_channel("analysis")
    ]

    output_messages = [
        Message.from_role_and_content(Role.ASSISTANT, output_str).with_channel("final")
    ]

    def _get_decoded_tokens(messages: list[Message]):
        tokens = encoding.render_conversation(Conversation.from_messages(messages))
        decoded = encoding.decode(tokens)
        return decoded

    decoded_tokens = _get_decoded_tokens(input_messages) + _get_decoded_tokens(output_messages)
    return json.dumps(decoded_tokens)

master_df = master_df.with_columns(
    pl.struct(["attack_prompt", "attack_reply"])
    .map_elements(render_harmony_conversation, return_dtype=pl.String)
    .alias("harmony_conversation")
)

master_df = master_df.with_columns(pl.arange(0, len(master_df)).alias("row_number"))
master_df = master_df.select(pl.col("row_number"), *[col for col in master_df.columns if col != "row_number"])

master_df.write_parquet("gpt-oss-red-teaming.parquet", compression="snappy")
