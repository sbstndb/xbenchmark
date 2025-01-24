NUMACTL="numactl --physcpubind 1" # pin the process on core 1
BENCHMARK_DIR="src"
RESULT_DIR="results"

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
    OUTPUT_FILE="$RESULTS_DIR/$TEST.json"

    if [[ -x "$BENCHMARK" ]]; then
        echo "  Lancement de $TEST..."
        "$BENCHMARK" --benchmark_format=json > "$OUTPUT_FILE"
        if [[ $? -eq 0 ]]; then
            echo "  Résultat sauvegardé : $OUTPUT_FILE"
        else
            echo "  Erreur lors de l'exécution de $TEST" >&2
        fi
    else
        echo "  Fichier $BENCHMARK introuvable ou non exécutable !" >&2
    fi
done



