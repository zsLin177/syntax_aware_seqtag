# Syntax Aware Seqtag

## Environment
    python3
    torch == 1.10
    transformers == 4.5.1

## Training 
* Train a simple seqtag model (without CRF)
```shell
python -m supar.cmds.seqtag train -b \
        --train data/ptb/ptb/train.conllx \
        --dev data/ptb/ptb/dev.conllx \
        --test data/ptb/ptb/test.conllx \
        --batch-size 3000 \
        --encoder transformer \
        --embed data/glove.6B.300d.txt \
        --feat char \
        -p exp/transformer-simple-pos/model \
        -d 4
```
-d: use which gpu

* Train a crf seqtag model
```shell
python -m supar.cmds.crf_seqtag train -b \
        --train data/ptb/ptb/train.conllx \
        --dev data/ptb/ptb/dev.conllx \
        --test data/ptb/ptb/test.conllx \
        --batch-size 3000 \
        --encoder transformer \
        --embed data/glove.6B.300d.txt \
        --feat char \
        -p exp/transformer-crf-pos/model \
        -d 4
```
* with bert
```shell
python -m supar.cmds.crf_seqtag train -b \
        --train data/ctb/ctb7/train.conll \
        --dev data/ctb/ctb7/dev.conll \
        --test data/ctb/ctb7/test.conll \
        --batch-size 3000 \
        --encoder bert \
        --bert ./bert-base-chinese \
        -p exp/debug/model \
        -d 4 
```

## Evaluate
```shell
python -m supar.cmds.seqtag evaluate --data data/ptb/ptb/test.conllx \
        -p exp/transformer-simple-pos/model \
        -d 4

python -m supar.cmds.crf_seqtag evaluate --data data/ptb/ptb/test.conllx \
        -p exp/transformer-crf-pos/model \
        -d 5
```

## Predict
```shell
python -m supar.cmds.seqtag predict --data data/ptb/ptb/test.conllx \
        --pred test.pred \
        -p exp/transformer-simple-pos/model \
        -d 4
python -m supar.cmds.crf_seqtag predict --data data/ptb/ptb/test.conllx \
        --pred test.pred \
        -p exp/transformer-crf-pos/model \
        -d 4
```

