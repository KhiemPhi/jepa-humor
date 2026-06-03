#!/usr/bin/env bash
SESSION="beir_eval"
WORKDIR="/home/tudao/jepa-humor"
CONFIG="configs/eval_beir_nq.yml"
CKPT_BASE="billionare_file/checkpoints_roberta_large/pretrain"

# window-name : gpu : ckpt (empty = vanilla)
declare -a JOBS=(
  "roberta_vanilla:4:"
  "roberta_mnli:5:$CKPT_BASE/best_pretrain_mnli_exp1_.pt"
  "roberta_rte:6:$CKPT_BASE/best_pretrain_rte_exp1_.pt"
  "roberta_qnli:7:$CKPT_BASE/best_pretrain_qnli_exp1_.pt"
)

tmux kill-session -t "$SESSION" 2>/dev/null

first=1
for job in "${JOBS[@]}"; do
  IFS=: read -r name gpu ckpt <<< "$job"

  if [ -z "$ckpt" ]; then
    ckpt_args=""
    exp="roberta_large_vanilla"
  else
    task="${name#roberta_}"
    ckpt_args="--model_ckpt $ckpt --is_pretrain_ckpt"
    exp="roberta_large_pretrain_${task}"
  fi

  cmd="cd $WORKDIR && python scripts/eval_beir.py --config $CONFIG $ckpt_args --gpu_id $gpu --experiment_name $exp; echo '✅ $name done'"

  if [ $first -eq 1 ]; then
    tmux new-session -d -s "$SESSION" -n "$name" -x 220 -y 50
    tmux send-keys -t "$SESSION:$name" "$cmd" Enter
    first=0
  else
    tmux new-window -t "$SESSION" -n "$name"
    tmux send-keys -t "$SESSION:$name" "$cmd" Enter
  fi
done

echo "Started tmux session '$SESSION' with 4 windows:"
echo "  roberta_vanilla (GPU 4)  roberta_mnli (GPU 5)  roberta_rte (GPU 6)  roberta_qnli (GPU 7)"
echo "Attach: tmux attach -t $SESSION"
