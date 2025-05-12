uv venv tina_env --python python3.10
source ./tina_env/bin/activate

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

uv pip install -r ./scripts/uvset/requirements_tina.txt

deactivate