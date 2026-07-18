# Ejecución de RaMem en Lightning AI

Esta es la secuencia canónica. Los datos se descargan dentro de Lightning; nunca se suben desde el
equipo local ni se incluyen en Git.

## 0. Preparación única fuera de Lightning

1. Copiar el contrato de variables y completar el `.env` local (nunca se publica):

```bash
cp .env.example .env
```

2. En Hugging Face, iniciar sesión y aceptar la licencia de:
   - `google/gemma-3-270m-it`
   - `google/gemma-3-1b-it`
3. Crear un token de Hugging Face con permiso de lectura y escribirlo en `HF_TOKEN` dentro de
   `.env`.
4. Crear el repositorio remoto y publicar el código:

```bash
git add .
git commit -m "Initialize RaMem real training pipeline"
git remote add origin <URL_DEL_REPOSITORIO>
git push -u origin main
```

No ejecutar `git add data/raw data/processed artifacts .cache`; están ignorados deliberadamente.

## 1. Crear el Studio sin consumir GPU

Crear un Lightning Studio nuevo y mantener inicialmente la máquina CPU gratuita. Si el Studio ya
tiene un `.env`, conservarlo y añadir las claves ausentes de `.env.example`. Como mínimo debe tener:

```text
HF_TOKEN=<token de lectura de Hugging Face>
HF_HOME=/teamspace/studios/this_studio/.cache/huggingface
HF_DATASETS_CACHE=/teamspace/studios/this_studio/.cache/huggingface/datasets
TOKENIZERS_PARALLELISM=true
RAMEM_CONFIG=configs/default.yaml
RAMEM_RAW_DATA_DIR=data/raw
RAMEM_PROCESSED_DATA_DIR=data/processed
RAMEM_ARTIFACTS_DIR=artifacts
RAMEM_TRAIN_CONFIG=configs/training/gemma_1b_grounded_qlora.yaml
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UV_LINK_MODE=copy
```

Abrir un terminal nuevo después de guardar los secretos y ejecutar:

```bash
git clone <URL_DEL_REPOSITORIO>
cd ramem
test -f ../.env || cp .env.example ../.env
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv sync --extra dev --extra training
uv run ramem doctor
```

La disposición recomendada en Lightning es:

```text
/home/zeus/content/.env
/home/zeus/content/ramem/       # raíz Git del proyecto
```

Completar `HF_TOKEN` en el `.env` padre si no fue inyectado como secreto. RaMem busca `.env` en la
raíz del repositorio y en sus directorios padres, y lo carga con
`override=false`: los secretos definidos por Lightning tienen prioridad y nunca son reemplazados por
el archivo. `ramem doctor` debe mostrar Python 3.12; la ausencia de GPU es correcta en esta fase CPU.

## 2. Descargar datos reales desde Lightning

Descargar únicamente SQuAD-es `train` para el primer ajuste grounded:

```bash
uv run ramem-download --dataset squad-es
```

El comando consulta el repositorio oficial, valida `CC-BY-4.0`, resuelve el commit inmutable,
descarga el split y genera `data/raw/download_manifest.json` con revisión y SHA-256 del árbol local.
Si la carpeta ya existe, el comando se detiene para proteger la inmutabilidad de datos raw.

Preparar prompts grounded usando solo filas reales:

```bash
uv run ramem-prepare
```

La salida queda en `data/processed/grounded-qa-es-v1`. Se deriva un 5% de validación mediante hash
estable del ID; no se usa el split público de validación.

Para los experimentos de retrieval E02–E05, descargar después y por separado:

```bash
uv run ramem-download --dataset miracl-es
uv run ramem-download --dataset miracl-corpus-es
```

El corpus es mucho mayor; no descargarlo antes de necesitar E02.

## 3. Smoke test en T4

Cambiar el Studio a una **T4 de 16 GB**, abrir un terminal nuevo y comprobar:

```bash
nvidia-smi
uv run ramem doctor
```

Ejecutar un entrenamiento corto con 64 filas reales y contexto 512:

```bash
tmux new-session -d -s ramem-train './scripts/train/lightning_t4_smoke.sh'
tmux attach -t ramem-train
```

Para separarse sin detener el proceso: `Ctrl-b`, después `d`. El log persistente queda en
`artifacts/training/t4-smoke.log` y puede seguirse con `tail -f`.

Debe aparecer:

```text
artifacts/training/gemma-1b-smoke/resolved_config.json
artifacts/training/gemma-1b-smoke/adapter-final/
```

Revisar que `resolved_target_modules` no esté vacío y que el entrenamiento/evaluación terminen sin
NaN ni OOM. El código obtiene los nombres de módulos del checkpoint real; no los asume ciegamente.

El trainer conserva como máximo los dos checkpoints más recientes. Cada checkpoint incluye el
estado necesario para reanudar el optimizador, scheduler, pasos y épocas. Si el comando se vuelve a
ejecutar con el mismo `output_dir`, detecta automáticamente el `checkpoint-N` numéricamente mayor y
continúa desde allí. No borrar esos directorios ni cambiar `output_dir` al reanudar.

## 4. Entrenamiento completo en L4

Detener la T4 y cambiar a una **L4 de 24 GB**. La configuración completa usa contexto 4096, QLoRA
NF4, gradient checkpointing, batch físico 1 y acumulación 32:

```bash
uv run ramem-train --config configs/training/gemma_1b_grounded_qlora.yaml
```

No cerrar ni eliminar el Studio. Lightning mantiene el entorno y archivos al detener la máquina.
Supervisar GPU y disco desde otro terminal:

```bash
watch -n 2 nvidia-smi
du -sh data/raw data/processed artifacts .cache/huggingface
```

## 5. Repeticiones científicas

La primera ejecución usa seed 42. Después de validar calidad y costo, duplicar la configuración con
seeds `13` y `2026`, cambiando también `output_dir`. No reutilizar el mismo directorio entre seeds.
Cada ejecución conserva configuración resuelta, GPU, dtype y módulos LoRA reales.

## 6. Reanudar un Studio

```bash
cd /home/zeus/content/ramem
git pull --ff-only
uv sync --extra dev --extra training
uv run ramem doctor
```

No repetir descargas si sus carpetas y `download_manifest.json` están presentes. Para actualizar un
dataset se debe usar una ruta versionada nueva; nunca sobrescribir `data/raw`.

## 7. Sincronización cotidiana por GitHub

Desde el equipo local, después de validar y crear un commit:

```bash
git push origin main
```

En Lightning, antes de ejecutar cualquier trabajo:

```bash
cd /home/zeus/content/ramem
git pull --ff-only origin main
```

No volver a transferir el proyecto con `scp`. Los datos y modelos permanecen únicamente en
Lightning y están excluidos por `.gitignore`.

## Orden resumido

```text
Aceptar Gemma → publicar Git → Studio CPU → instalar → descargar → preparar
→ T4 smoke test → revisar artefactos → L4 entrenamiento completo → seeds adicionales
```
