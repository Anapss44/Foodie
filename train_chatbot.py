"""Simple training helper to fine-tune a causal LM on chat-style data.

Usage:
  python train_chatbot.py --data data/chat_logs.jsonl --model gpt2 --output_dir ./fine-tuned-chatbot

The script expects `data/chat_logs.jsonl` to contain one JSON object per line.
Each object may be either:
  - {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
or
  - {"prompt": "...", "response": "..."}

The trainer will construct prompt/response pairs and fine-tune a causal LM using the Hugging Face Trainer.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to JSONL chat logs")
    p.add_argument("--model", default="gpt2", help="Base model name or path")
    p.add_argument("--output_dir", default="./fine-tuned-chatbot")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    return p.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_pairs_from_messages(obj: Dict) -> List[Dict]:
    """Given an object with `messages` list, extract prompt/response pairs.
    For each assistant message, the prompt is the preceding user messages since last assistant.
    """
    out = []
    messages = obj.get("messages") or []
    buffer = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        if not content:
            continue
        if role == "assistant":
            if buffer:
                prompt = "\n".join(buffer)
            else:
                prompt = ""
            out.append({"prompt": prompt, "response": content})
            buffer = []
        else:
            buffer.append(content)
    return out


def prepare_examples(data_path: str) -> List[Dict]:
    items = load_jsonl(data_path)
    examples = []
    for obj in items:
        if "messages" in obj:
            pairs = build_pairs_from_messages(obj)
            examples.extend(pairs)
        elif "prompt" in obj and "response" in obj:
            examples.append({"prompt": obj["prompt"], "response": obj["response"]})
    return examples


def make_dataset_and_train(args):
    try:
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
        import torch
    except Exception as exc:
        print("Missing training dependencies. Install 'datasets', 'transformers', and 'torch'.")
        raise

    examples = prepare_examples(args.data)
    if not examples:
        raise SystemExit("No training examples found in data file.")

    # Prepare texts: concat prompt + response with a separator.
    sep = "\n"  # keep simple
    texts = [((e.get("prompt") or "") + sep + (e.get("response") or "")).strip() for e in examples]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=1024)

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    print("Training complete. Model saved to", str(output_dir))


def main():
    args = parse_args()
    if not os.path.exists(args.data):
        raise SystemExit(f"Data file not found: {args.data}")
    make_dataset_and_train(args)


if __name__ == "__main__":
    main()
