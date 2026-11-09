"""Copia de scripts/entrenamiento.py para GPU.

Cambios respecto al original (el resto es idéntico para poder comparar):
  - sin torch.set_num_threads(6) ni use_cpu=True;
  - bf16 + TF32 si la GPU lo admite; batch de evaluación 128 (no cambia resultados);
  - parámetro `seed` (por defecto SEED) para entrenar con varias semillas.
"""
import time

import numpy as np
import pandas as pd
import torch
import transformers
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

from common_jb import MODELOS, SEED, TAREAS, tokenizar

GPU = torch.cuda.is_available()
BF16 = GPU and torch.cuda.is_bf16_supported()
if GPU:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def entorno() -> dict:
    return {"gpu": torch.cuda.get_device_name(0) if GPU else "cpu", "bf16": BF16,
            "torch": torch.__version__, "transformers": transformers.__version__}


def a_dataset(tokenizer, df: pd.DataFrame) -> Dataset:
    enc = tokenizar(tokenizer, df)
    enc["labels"] = df["labels"].tolist()
    return Dataset.from_dict(dict(enc))


def predecir(model, tokenizer, df: pd.DataFrame, batch: int = 128) -> np.ndarray:
    """Predicción en el mismo orden que df (trainer.predict puede reordenar los ejemplos)."""
    model.eval()
    dispositivo = next(model.parameters()).device
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
        for i in range(0, len(df), batch):
            enc = tokenizer.pad(tokenizar(tokenizer, df.iloc[i : i + batch]), return_tensors="pt").to(dispositivo)
            preds.append(model(**enc).logits.float().argmax(-1).cpu())
    return torch.cat(preds).numpy()


def pesos_clase(train: pd.DataFrame, n_clases: int) -> torch.Tensor:
    conteo = np.bincount(train["labels"], minlength=n_clases)
    return torch.tensor(len(train) / (n_clases * conteo), dtype=torch.float)


def metricas(pred):
    y_pred = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, y_pred),
            "f1_macro": f1_score(pred.label_ids, y_pred, average="macro")}


def entrenar(tarea: str, modelo: str, train: pd.DataFrame, dev: pd.DataFrame, lr: float, batch: int,
             epochs: int, output_dir, guardar: bool, paciencia: int | None = None, seed: int = SEED):
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
            logits = outputs.logits.float()
            return loss_fn.to(logits.device)(logits, labels)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=128,
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
        bf16=BF16,
        report_to="none",
        seed=seed,
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
