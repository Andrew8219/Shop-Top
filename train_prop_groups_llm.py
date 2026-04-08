#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, random_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ========================
#   ЗАГРУЗКА РАЗМЕТКИ
# ========================

def load_labeled_records(
    path: str = "data/prop_group_clusters_for_labeling.jsonl",
) -> List[Dict]:
    """
    Поддерживает два формата:
      1) Один большой JSON-массив: [ {...}, {...}, ... ]
      2) Поток отдельных JSON-объектов (в т.ч. многострочных), как задумывалось ранее.
    """
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()

    records: List[Dict] = []

    # --- Вариант 1: файл — это JSON-массив ---
    if stripped.startswith("["):
        raw = json.loads(text)
        if isinstance(raw, list):
            for rec in raw:
                if not isinstance(rec, dict):
                    continue
                label = (rec.get("group_label") or "").strip()
                names = rec.get("names") or []
                if label and names:
                    records.append(rec)
        LOG.info("Загружено размеченных кластеров (формат: массив): %d", len(records))
        return records

    # --- Вариант 2: старый потоковый формат (JSONL / подряд объекты) ---
    buf = ""
    depth = 0

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue

        buf += raw_line
        depth += raw_line.count("{") - raw_line.count("}")

        if depth == 0 and buf.strip():
            rec = json.loads(buf)
            label = (rec.get("group_label") or "").strip()
            names = rec.get("names") or []
            if label and names:
                records.append(rec)
            buf = ""

    LOG.info("Загружено размеченных кластеров (формат: поток объектов): %d", len(records))
    return records


def build_examples(
    records: List[Dict],
) -> Tuple[List[str], List[str]]:
    """
    Из каждого кластера делаем по примеру на каждое имя свойства:
      text  = само имя свойства
      label = group_label кластера
    """
    texts: List[str] = []
    labels: List[str] = []

    for rec in records:
        label = (rec.get("group_label") or "").strip()
        if not label:
            continue
        names: List[str] = rec.get("names") or []
        for n in names:
            n = (n or "").strip()
            if not n:
                continue
            texts.append(n)
            labels.append(label)

    LOG.info("Всего обучающих примеров (имён свойств): %d", len(texts))
    return texts, labels


# ========================
#   DATASET
# ========================

class PropGroupClsDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        label_ids: List[int],
        tokenizer,
        max_length: int = 64,
    ) -> None:
        assert len(texts) == len(label_ids)
        self.tokenizer = tokenizer
        self.texts = texts
        self.label_ids = label_ids
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label_id = self.label_ids[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(label_id, dtype=torch.long)
        return item


# ========================
#   CLI / MAIN
# ========================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Дообучение rubert-tiny2 для группировки имён характеристик (mini-LLM)."
    )
    p.add_argument(
        "--data",
        default="data/prop_group_clusters_for_labeling.jsonl",
        help="Файл разметки кластеров (JSON-массив или поток JSON-объектов).",
    )
    p.add_argument(
        "--output",
        default="models/prop_groups_llm",
        help="Куда сохранить tokenizer + веса для inference в normalization.py.",
    )
    p.add_argument(
        "--base-model",
        default="cointegrated/rubert-tiny2",
        help="Базовая модель HuggingFace.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Число эпох (по умолчанию: 8; с --full: 120).",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Долгое обучение как в старой версии скрипта (120 эпох).",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Размер батча train/eval.")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    p.add_argument("--seed", type=int, default=42, help="Seed для train/val split.")
    p.add_argument(
        "--checkpoint-dir",
        default="models/prop_groups_llm_checkpoints",
        help="Промежуточные чекпоинты Trainer.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    data_path = args.data
    base_model_name = args.base_model
    output_dir = args.output

    if args.full:
        num_epochs = args.epochs if args.epochs is not None else 120
    else:
        num_epochs = args.epochs if args.epochs is not None else 8

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = load_labeled_records(data_path)
    if not records:
        LOG.error(
            "Нет размеченных кластеров (group_label пуст). "
            "Сначала заполни файл %s", data_path
        )
        return

    texts, label_strs = build_examples(records)

    # словарь метка -> id
    unique_labels = sorted(set(label_strs))
    label2id: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label: Dict[int, str] = {i: lbl for lbl, i in label2id.items()}

    LOG.info("Различных групп (классов): %d", len(unique_labels))
    for i, lbl in id2label.items():
        LOG.info("  %2d -> %s", i, lbl)

    label_ids = [label2id[lbl] for lbl in label_strs]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(unique_labels),
    )
    # пропишем маппинги, чтобы потом MiniLLMGroupClassifier их прочитал
    model.config.id2label = {int(i): lbl for i, lbl in id2label.items()}
    model.config.label2id = {lbl: int(i) for lbl, i in label2id.items()}

    dataset = PropGroupClsDataset(texts, label_ids, tokenizer)

    # простое train/val-разбиение
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(int(args.seed))
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    LOG.info("Train size: %d, Val size: %d", train_size, val_size)
    LOG.info("Эпох: %d (--full даёт 120 по умолчанию)", num_epochs)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=num_epochs,
        logging_steps=max(1, min(20, len(train_ds) // max(args.batch_size, 1))),
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=int(args.seed),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    LOG.info("Начинаем обучение…")
    trainer.train()

    LOG.info("Оцениваем на валидации…")
    try:
        metrics = trainer.evaluate()
        LOG.info("Val metrics: %s", metrics)
    except Exception as e:
        LOG.warning("Ошибка при evaluate(): %s", e)


    LOG.info("Сохраняем финальную модель в %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    LOG.info("Готово.")


if __name__ == "__main__":
    main()
