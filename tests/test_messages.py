import os
import json

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

def test_loss_mask(messages):
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]

    stop_reason = "complete"
    prompt_ids_list = []
    response_ids_list = []
    trajectory_ids_list = []
    loss_mask = []
    past_trajectory_len = 0
    # observation_len = 0

    # idx: 1
    # initial_input_ids_len: 4154
    # past_trajectory_len: 16568
    # past_response_len: 12414
    # current_prompt_len: 15565
    # current_response_len: 807
    # past_response_observation_len: 0
    # max_response_tokens = max_tokens + max_input_length - initial_input_len
    for idx, message in enumerate(token_messages):
        current_prompt_ids = message["prompt_token_ids"]
        current_response_ids = message["response_token_ids"]

        prompt_ids_list.append(current_prompt_ids)
        response_ids_list.append(current_response_ids)
        trajectory_ids_list.append(current_prompt_ids + current_response_ids)

        if idx == 0:
            # 4154
            initial_input_ids = current_prompt_ids
            initial_input_len = len(initial_input_ids)
            # 12414
            loss_mask = [1] * len(current_response_ids)
            continue

        past_trajectory_len = len(trajectory_ids_list[idx-1])
        past_response_len = len(response_ids_list[idx-1])
        current_prompt_len = len(current_prompt_ids)
        # 807
        current_response_len = len(current_response_ids)

        print("idx:", idx)
        print("initial_input_ids_len:", initial_input_len)
        print("past_trajectory_len:", past_trajectory_len)
        print("past_response_len:", past_response_len)
        print("current_prompt_len:", current_prompt_len)
        print("current_response_len:", current_response_len)

        # past_prompt_len = len(prompt_ids_list[idx-1]) if idx > 0 else 0
        past_response_observation_ids = current_prompt_ids[past_trajectory_len:]
        past_response_observation_len = len(past_response_observation_ids)
        print("past_response_observation_len:", past_response_observation_len)
        loss_mask.extend([0] * past_response_observation_len)
        loss_mask.extend([1] * current_response_len)
    
    response_ids = current_prompt_ids[initial_input_len:] + current_response_ids

    # if len(response_ids) >= max_response_tokens:
    #     response_ids = response_ids[:max_response_tokens]
    #     loss_mask = loss_mask[:max_response_tokens]
    #     stop_reason = "length"

    with open("debug_response_ids.txt", "w") as f:
        for mask, rid in zip(loss_mask, response_ids):
            f.write(f"Token ID: {rid}, Mask: {mask}, Token: {tokenizer.decode([rid])}\n")

    assert len(response_ids) == len(loss_mask), f"Response ids length {len(response_ids)} != loss mask length {len(loss_mask)}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages_path", type=str, required=True, help="Path to the messages JSON file")
    args = parser.parse_args()

    messages_files = os.listdir(args.messages_path)
    for file_name in messages_files:
        if not file_name.endswith(".json"):
            continue
        print(f"Processing file: {file_name}")
        full_path = os.path.join(args.messages_path, file_name)
        with open(full_path, "r") as f:
            messages = json.load(f)["messages"]

        try:
            test_loss_mask(messages)
        except Exception as e:
            print(f"Error processing {full_path}: {e}")

        # break