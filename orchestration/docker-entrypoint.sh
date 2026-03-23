#!/bin/bash
# M77: Export PYTHONHASHSEED in container entrypoint
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export NUMBA_DISABLE_JIT=1

echo "R4-MM-Clinical Pipeline"
echo "PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

exec "$@"
