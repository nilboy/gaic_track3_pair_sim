#!/bin/bash

export TOKENIZERS_PARALLELISM=True
model_name_o=$1
manual_seed=124525601
model_number_id=1
case $model_name_o in

  "bert-base")
    model_name="bert-base"
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=60
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    train_fgm=False
    manual_seed=124525601
    ;;
  "bert-base2")
    model_name="bert-base"
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=60
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    model_number_id=2
    train_fgm=False
    manual_seed=124525602
    ;;
  "bert-base-fgm")
    model_name="bert-base"
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=60
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    train_fgm=True
    manual_seed=124525603
    ;;
  "macbert-base")
    model_name="macbert-base"
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=60
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    train_fgm=False
    manual_seed=124525604
    ;;
  "macbert-base2")
    model_name="macbert-base"
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=60
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    model_number_id=2
    train_fgm=False
    manual_seed=124525605
    ;;
  "macbert-base-fgm")
    model_name="macbert-base"
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=60
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    train_fgm=True
    manual_seed=124525606
    ;;
  "bert-large")
    model_name="bert-large"
    model_type="bert"
    batch_size=96
    gradient_accumulation_steps=6
    num_epochs=50
    learning_rate=1.0e-4
    min_learning_rate=4e-5
    train_fgm=False
    manual_seed=124525607
    ;;
  "macbert-large")
    model_name="macbert-large"
    model_type="bert"
    batch_size=96
    gradient_accumulation_steps=6
    num_epochs=50
    learning_rate=1.0e-4
    min_learning_rate=4e-5
    train_fgm=False
    manual_seed=124525608
    ;;
  "roberta-base")
    model_name="roberta-base"
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=60
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    train_fgm=False
    manual_seed=124525609
    ;;
  "roberta-base-fgm")
    model_name="roberta-base"
    model_type="bert"
    batch_size=256
    gradient_accumulation_steps=2
    num_epochs=60
    learning_rate=1.5e-4
    min_learning_rate=4e-5
    train_fgm=True
    manual_seed=124525610
    ;;
  "roberta-large")
    model_name="roberta-large"
    model_type="bert"
    batch_size=96
    gradient_accumulation_steps=6
    num_epochs=50
    learning_rate=1.0e-4
    min_learning_rate=4e-5
    train_fgm=False
    manual_seed=124525611
    ;;
esac
echo "train mlm base model..."
python train/train_mlm.py --model_name=$model_name \
                          --model_type=$model_type \
                          --batch_size=$batch_size \
                          --gradient_accumulation_steps=$gradient_accumulation_steps \
                          --num_epochs=$num_epochs \
                          --learning_rate=$learning_rate \
                          --min_learning_rate=$min_learning_rate \
                          --use_fgm=$train_fgm \
                          --fgm_epsilon=0.3 \
                          --manual_seed=$manual_seed \
                          --model_number_id=$model_number_id
