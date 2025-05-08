uv venv tina_eval --python python3.11
source tina_eval/bin/activate

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

uv pip install -r requirements_tinaeval.txt

deactivate