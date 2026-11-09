import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from common import MODELOS, SEED, TAREAS, tokenizar

torch.set_num_threads(6)


def a_dataset(tokenizer, df: pd.DataFrame) -> Dataset:
    enc = tokenizar(tokenizer, df)
    enc["labels"] = df["labels"].tolist()
    return Dataset.from_dict(dict(enc))


def pesos_clase(train: pd.DataFrame, n_clases: int) -> torch.Tensor:
    conteo = np.bincount(train["labels"], minlength=n_clases)
    return torch.tensor(len(train) / (n_clases * conteo), dtype=torch.float)


def metricas(pred):
    y_pred = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, y_pred),
            "f1_macro": f1_score(pred.label_ids, y_pred, average="macro")}


def entrenar(tarea: str, modelo: str, train: pd.DataFrame, dev: pd.DataFrame, lr: float, batch: int,
             epochs: int, output_dir, guardar: bool, paciencia: int | None = None):
    """Entrena y devuelve (trainer, historial de evaluación en dev por época, segundos por ejemplo)."""
    clases = TAREAS[tarea]["clases"]
    tokenizer = AutoTokenizer.from_pretrained(MODELOS[modelo])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELOS[modelo], num_labels=len(clases),
        id2label=dict(enumerate(clases)), label2id={c: i for i, c in enumerate(clases)},
    )

    compute_loss_func = None
    if tarea == "definitud":
        loss_fn = torch.nn.CrossEntropyLoss(weight=pesos_clase(train, len(clases)))

        def compute_loss_func(outputs, labels, num_items_in_batch=None):
            return loss_fn(outputs.logits, labels)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        warmup_steps=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch" if guardar else "no",
        save_total_limit=1,
        load_best_model_at_end=guardar,
        metric_for_best_model="f1_macro",
        logging_strategy="epoch",
        train_sampling_strategy="group_by_length",
        use_cpu=True,
        report_to="none",
        seed=SEED,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=paciencia)] if guardar and paciencia else []
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=a_dataset(tokenizer, train),
        eval_dataset=a_dataset(tokenizer, dev),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metricas,
        compute_loss_func=compute_loss_func,
        callbacks=callbacks,
    )
    inicio = time.time()
    salida = trainer.train()
    duracion = time.time() - inicio
    historial = [
        {"epoch": round(h["epoch"]), "dev_f1_macro": h["eval_f1_macro"], "dev_accuracy": h["eval_accuracy"]}
        for h in trainer.state.log_history if "eval_f1_macro" in h
    ]
    seg_por_ejemplo = duracion / (len(train) * salida.metrics.get("epoch", epochs))
    return trainer, historial, seg_por_ejemplo
