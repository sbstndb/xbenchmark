NUMACTL="numactl --physcpubind 1" # pin the process on core 1
BENCHMARK_DIR="src"
RESULTS_DIR="results"

PYTHON_PLOT="../src/python/plot.py"
PYTHON_ARGS="--logx --logy"


TESTS=("allocation/allocation"
	"blas1/blas1_complex_operation"
	"blas1/blas1_fma"
	"blas1/blas1_simple_operation"
	"find/find"
	"op/op"
	"view/view_all"
	"view/view_stride"
)



mkdir -p $RESULT_DIR

echo "Lancement des benchmarks ..."
for TEST in "${TESTS[@]}"; do
    BENCHMARK="$BENCHMARK_DIR/$TEST"
    OUTPUT_NAME=${TEST#*/}
    OUTPUT_FILE="$RESULTS_DIR/$OUTPUT_NAME.json"



    echo "$BENCHMARK"
    if [[ -x "$BENCHMARK" ]]; then
        echo "  Lancement de $TEST..."
        "$BENCHMARK" --benchmark_out="${OUTPUT_FILE}"

	python $PYTHON_PLOT -f ${OUTPUT_FILE} ${PYTHON_ARGS} --output "${OUTPUT_NAME}.png"

        if [[ $? -eq 0 ]]; then
            echo "  Résultat sauvegardé : $OUTPUT_FILE"
        else
            echo "  Erreur lors de l'exécution de $TEST" >&2
        fi
    else
        echo "  Fichier $BENCHMARK introuvable ou non exécutable !" >&2
    fi
done



