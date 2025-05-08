for EXCLUDE in {1..9}; do
  LOGDIR="exp_${EXCLUDE}/cifar10"
  mkdir -p "$LOGDIR"  # Create the log directory if it doesn't exist

  for EXPID in {0..15}; do
    echo "Running EXCLUDE=$EXCLUDE, EXPID=$EXPID"
    CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py \
      --dataset=cifar10 \
      --epochs=100 \
      --save_steps=20 \
      --arch wrn28-2 \
      --num_experiments 16 \
      --expid $EXPID \
      --logdir "$LOGDIR" \
      --exclude_class $EXCLUDE
  done
done
