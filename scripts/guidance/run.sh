ORACLE=SARSCoV2Beta
ORACLE_SEED_IDX=0
MAX_MUTATIONS=3
# MASK_REGION_FLAG="--mask-region CDR_overall"
MASK_REGION_FLAG=""
SEED=42
METHOD=$1
TESTING=false

# Use Custom Starting Seed Antibody Sequence (leave empty to use oracle seed by index)
# If SEED_SEQ is set, it will override ORACLE_SEED_IDX
SEED_SEQ=QVQLVQSGAEVKKPGASVKVSCKASGYTFTTYAMHWVRQAPGQRLEWMGWINAGNGNTKYSQKFQGRVTITRDTSASTAYMELSSLRSEDTAVYYCAGGGGRRLQFDYFDYWGQGTLVTV
# Example custom sequences (uncomment and set SEED_SEQ to one of these to use):
# SEED_HV_CHAIN=QVQLVQSGAEVKKPGASVKVSCKASGYTFTTYAMHWVRQAPGQRLEWMGWINAGNGNTKYSQKFQGRVTITRDTSASTAYMELSSLRSEDTAVYYCAGGGGRRLQFDYFDYWGQGTLVTV
# SEED_LT_CHAIN=DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYDASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNGYPWTFGQGTKV

if [ -z "$METHOD" ]; then
    echo "Usage: $0 <method>"
    echo "Available methods: cosine, hotflip, genetic, mlm_esm2, mlm_ablang, random"
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
    FINAL_N=100
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

# Build seed sequence argument if SEED_SEQ is provided
if [ -n "$SEED_SEQ" ]; then
    SEED_SEQ_ARG="--seed-seq \"$SEED_SEQ\""
else
    SEED_SEQ_ARG=""
fi

case $METHOD in
    cosine)
        eval "uv run python $DEBUG_FLAG cosine.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            $SEED_SEQ_ARG \
            --max-mutations $MAX_MUTATIONS \
            $MASK_REGION_FLAG \
            --batch-size $FINAL_N \
            --use-guided \
            --branch-length 0.025 \
            --use-discrete-steps \
            --guidance-strength 0.2 \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    hotflip)
        eval "uv run python $DEBUG_FLAG hotflip.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            $SEED_SEQ_ARG \
            --max-mutations $MAX_MUTATIONS \
            --top-k $TOP_K_SITES \
            --batch-size $FINAL_N \
            $MASK_REGION_FLAG \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    genetic)
        eval "uv run python $DEBUG_FLAG genetic.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            $SEED_SEQ_ARG \
            --pop-size $FINAL_N \
            --top-k $TOP_K \
            --generations $N_GENERATIONS \
            --max-mutations $MAX_MUTATIONS \
            --mutation-rate 1 \
            $MASK_REGION_FLAG \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    mlm_esm2)
        eval "uv run python $DEBUG_FLAG mlm.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            $SEED_SEQ_ARG \
            --mlm-model esm2_t30_150M_UR50D \
            --guidance-strength 1.0 \
            --sweeps $MAX_MUTATIONS \
            --use-local-fitness \
            --max-mutations $MAX_MUTATIONS \
            $MASK_REGION_FLAG \
            --batch-size $FINAL_N \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    mlm_ablang)
        eval "uv run python $DEBUG_FLAG mlm.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            $SEED_SEQ_ARG \
            --mlm-model ablang \
            --guidance-strength 10.0 \
            --sweeps $MAX_MUTATIONS \
            --use-local-fitness \
            --max-mutations $MAX_MUTATIONS \
            $MASK_REGION_FLAG \
            --batch-size $FINAL_N \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    random)
        eval "uv run python $DEBUG_FLAG random_baseline.py \
            --oracle $ORACLE \
            --oracle-seed-idx $ORACLE_SEED_IDX \
            $SEED_SEQ_ARG \
            --max-mutations $MAX_MUTATIONS \
            $MASK_REGION_FLAG \
            --batch-size $FINAL_N \
            --seed $SEED \
            $SAVE_CSV_FLAG \
            $LOG_REDIRECT"
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Available methods: cosine, hotflip, genetic, mlm_esm2, mlm_ablang, random"
        exit 1
        ;;
esac