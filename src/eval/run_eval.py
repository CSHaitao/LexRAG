import os
import logging
import json
import argparse
from metrics import UnifiedEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_evaluation_only(output_dir, response_file):
    logging.info("[1/1] Running evaluation...")

    evaluator = UnifiedEvaluator()
    metrics_per_turn = {}

    id_to_response = {}
    with open(response_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            id_to_response[record["id"]] = record["response"]

    for turn_num in sorted(os.listdir(output_dir)):
        input_path = os.path.join(output_dir, turn_num)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_preds = []
        valid_refs = []
        valid_keywords = []

        for sample in data:
            sample_id = sample["id"]
            pred_response = id_to_response.get(sample_id, "").strip()

            if pred_response:
                valid_preds.append(pred_response)
                valid_refs.append(sample["reference"])
                valid_keywords.append(sample["keywords"])

        metrics = evaluator.calculate_all_metrics(valid_preds, valid_refs, valid_keywords)
        metrics_per_turn[turn_num] = metrics

        logging.info(f"\n{turn_num.upper()} Metrics:")
        for k, v in metrics.items():
            logging.info(f"{k.ljust(15)}: {v:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Metrics")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="generated samples"
    )
    parser.add_argument(
        "--response_file", type=str, required=True, help="generated responses"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_evaluation_only(args.output_dir, args.response_file)
