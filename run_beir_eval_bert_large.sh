#!/usr/bin/env bash
SESSION="beir_eval_bert"
WORKDIR="/home/tudao/jepa-humor"
CONFIG="configs/eval_beir_nq_bert_large.yml"
CKPT_BASE="billionare_file/checkpoints_better_bert_large/pretrain"

# window-name : gpu : ckpt (empty = vanilla)
declare -a JOBS=(
  "bert_vanilla:4:"
  "bert_mnli:5:$CKPT_BASE/best_pretrain_mnli_exp1_.pt"
  "bert_rte:6:$CKPT_BASE/best_pretrain_rte_exp1_.pt"
  "bert_qnli:7:$CKPT_BASE/best_pretrain_qnli_exp1_.pt"
)

tmux kill-session -t "$SESSION" 2>/dev/null

first=1
for job in "${JOBS[@]}"; do
  IFS=: read -r name gpu ckpt <<< "$job"

  if [ -z "$ckpt" ]; then
    ckpt_args=""
    exp="bert_large_vanilla"
  else
    task="${name#bert_}"
    ckpt_args="--model_ckpt $ckpt --is_pretrain_ckpt"
    exp="bert_large_pretrain_${task}"
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
echo "  bert_vanilla (GPU 4)  bert_mnli (GPU 5)  bert_rte (GPU 6)  bert_qnli (GPU 7)"
echo "Attach: tmux attach -t $SESSION"
