#!/bin/bash
model_name=$1
case $model_name in

  "bert-base")
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=80
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    ;;
  "nezha-base")
    model_type="nezha"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=80
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    ;;
  "macbert-base")
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=80
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    ;;
  "bert-large")
    model_type="bert"
    batch_size=96
    gradient_accumulation_steps=6
    num_epochs=80
    learning_rate=1.2e-4
    min_learning_rate=4e-5
    ;;
  "nezha-large")
    model_type="nezha"
    batch_size=96
    gradient_accumulation_steps=6
    num_epochs=80
    learning_rate=1.2e-4
    min_learning_rate=4e-5
    ;;
  "macbert-large")
    model_type="bert"
    batch_size=96
    gradient_accumulation_steps=6
    num_epochs=80
    learning_rate=1.2e-4
    min_learning_rate=4e-5
    ;;
esac
echo "train mlm base model..."
python train/train_mlm.py --model_name=$model_name \
                          --model_type=$model_type \
                          --batch_size=$batch_size \
                          --gradient_accumulation_steps=$gradient_accumulation_steps \
                          --num_epochs=$num_epochs \
                          --learning_rate=$learning_rate \
                          --min_learning_rate=$min_learning_rate
echo "train mlm fgm model..."
python train/train_mlm.py --model_name=$model_name \
                          --model_type=$model_type \
                          --batch_size=$batch_size \
                          --gradient_accumulation_steps=$gradient_accumulation_steps \
                          --num_epochs=$num_epochs \
                          --learning_rate=$learning_rate \
                          --min_learning_rate=$min_learning_rate \
                          --use_fgm=True \
                          --fgm_epsilon=0.4
echo "train classification..."
for data_id in {0..4}
  do
    python train/train_best_pipeline.py --data_id=$data_id \
                                        --model_names=$model_name \
                                        --model_num=4
  done
