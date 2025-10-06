"""Convert span format to QA training format.

Convert from our format:
{
  "text": "...",
  "properties": [
    {
      "id": "marchio",
      "question": "Qual è il marchio?",
      "answer": {"text": "Knauf", "start": 290, "end": 295}
    }
  ]
}

To QA training format:
{
  "id": "record_1_marchio",
  "context": "...",
  "question": "Qual è il marchio?",
  "answers": [{"text": "Knauf", "start": 290}],
  "property_id": "marchio"
}
"""
import json
from pathlib import Path


def convert_to_qa_format(input_file: Path, output_file: Path):
    """Convert span format to QA training format."""

    examples = []
    record_idx = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            record_idx += 1

            text = record["text"]

            # Convert each property to a separate QA example
            for prop in record.get("properties", []):
                example = {
                    "id": f"record_{record_idx}_{prop['id']}",
                    "context": text,
                    "question": prop["question"],
                    "answers": [{
                        "text": prop["answer"]["text"],
                        "start": prop["answer"]["start"]
                    }],
                    "property_id": prop["id"]
                }
                examples.append(example)

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    return len(examples)


def main():
    base_dir = Path(__file__).parent.parent.parent

    # Convert train
    train_in = base_dir / "resources/data/train/span/train_qa.jsonl"
    train_out = base_dir / "resources/data/train/span/train_qa_format.jsonl"

    # Convert val
    val_in = base_dir / "resources/data/train/span/val_qa.jsonl"
    val_out = base_dir / "resources/data/train/span/val_qa_format.jsonl"

    print("Converting to QA training format...")

    train_count = convert_to_qa_format(train_in, train_out)
    print(f"Train examples: {train_count}")
    print(f"  Written to: {train_out}")

    val_count = convert_to_qa_format(val_in, val_out)
    print(f"Val examples: {val_count}")
    print(f"  Written to: {val_out}")

    # Show sample
    print("\nSample example:")
    with open(train_out, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
