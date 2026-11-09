# Rama `jb`: experimentos en GPU (clúster HU, lustre)

Repite los experimentos de `main` en GPU sin tocar el trabajo en PC:

- `scripts_jb/`: copias `_jb` de los scripts. Leen los mismos `dataset/processed/` y escriben en `resultados_jb/`.
  - `entrenamiento_jb.py`: igual que `entrenamiento.py`, pero sin `use_cpu`/`set_num_threads`, con bf16/TF32 en GPU y con parámetro `seed`.
  - `02_baseline_jb.py` y `03_small_test_jb.py`: las mismas líneas base y la misma rejilla (lr × batch {8, 16} × 5 épocas, 40 % de train).
  - `04_final_jb.py` (nuevo): entrena en train completo con los hparams del 03 y 3 semillas (42, 43, 44), elige la época por F1 macro en dev y evalúa en test.
  - `05_subir_hf_jb.py` (nuevo): sube el modelo final (la semilla con mejor dev) a HF, en un repo privado.
- `slurm_jb/`: `env_jb.sh` y arrays de SLURM (10 pares = 2 tareas × 5 modelos).
- `resultados_jb/{mdo,definitud}/`: `baseline/`, `tiempo_*.json`, `hparams_*.json`, `final_*/metricas.json` y predicciones.

```
source slurm_jb/env_jb.sh
$PY scripts_jb/02_baseline_jb.py --task mdo
J=$(sbatch --parsable slurm_jb/03_busqueda_jb.sbatch)
sbatch --dependency=aftercorr:$J slurm_jb/04_final_jb.sbatch
HF_TOKEN=$(cat $MJ/.hf_token_write) $PY scripts_jb/05_subir_hf_jb.py
```

En GPU con bf16 los resultados no son idénticos bit a bit a los de CPU, pero sí comparables.
