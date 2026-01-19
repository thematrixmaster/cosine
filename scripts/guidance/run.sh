ORACLE=SARSCoV1
ORACLE_SEED_IDX=0
MAX_MUTATIONS=5
MASK_REGION=CDR_overall
SEED=42
METHOD=$1
TESTING=false

if [ -z "$METHOD" ]; then
    echo "Usage: $0 <method>"
    echo "Available methods: cosine, hotflip, genetic, mlm_esm2, mlm_ablang"
    exit 1
fi

# Set debug flag if TESTING is true
if [ "$TESTING" = true ]; then
    DEBUG_FLAG="-m pdb"
    SAVE_CSV_FLAG=""
    FINAL_N=10
    LOG_REDIRECT=""
else
    DEBUG_FLAG=""
    SAVE_CSV_FLAG="--save-csv"
    FINAL_N=1000
    # Create logs directory if it doesn't exist
    mkdir -p logs
    # Create timestamped log file
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/${METHOD}_${ORACLE}_seed${ORACLE_SEED_IDX}_${TIMESTAMP}.log"
    LOG_REDIRECT="2>&1 | tee $LOG_FILE"
fi

# set top-k to final_n // 2
TOP_K=$(($FINAL_N / 2))
TOP_K_SITES=15
N_GENERATIONS=9

case $METHOD in
    cosine)
        eval "uv run python $DEBUG_FLAG cosine.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            --max-mutations $MAX_MUTATIONS \
            --mask-region $MASK_REGION \
            --batch-size $FINAL_N \
            --use-guided \
            --branch-length 0.035 \
            --use-discrete-steps \
            --guidance-strength 20.0 \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    hotflip)
        eval "uv run python $DEBUG_FLAG hotflip.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            --max-mutations $MAX_MUTATIONS \
            --top-k $TOP_K_SITES \
            --batch-size $FINAL_N \
            --mask-region $MASK_REGION \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    genetic)
        eval "uv run python $DEBUG_FLAG genetic.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            --pop-size $FINAL_N \
            --top-k $TOP_K \
            --generations $N_GENERATIONS \
            --max-mutations $MAX_MUTATIONS \
            --mutation-rate 1 \
            --mask-region $MASK_REGION \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    mlm_esm2)
        eval "uv run python $DEBUG_FLAG mlm.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            --mlm-model esm2_t30_150M_UR50D \
            --guidance-strength 50.0 \
            --sweeps 5 \
            --use-local-fitness \
            --max-mutations $MAX_MUTATIONS \
            --mask-region $MASK_REGION \
            --batch-size $FINAL_N \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    mlm_ablang)
        eval "uv run python $DEBUG_FLAG mlm.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            --mlm-model ablang \
            --guidance-strength 50.0 \
            --sweeps 5 \
            --use-local-fitness \
            --max-mutations $MAX_MUTATIONS \
            --mask-region $MASK_REGION \
            --batch-size $FINAL_N \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Available methods: cosine, hotflip, genetic, mlm_esm2, mlm_ablang"
        exit 1
        ;;
esac