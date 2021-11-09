# Syntax Aware Seqtag

## Pre-process

* use preprocess.py to produce data in form of conllu
    source file "data.txt.char.train.ner"
    target file "train.conllu"
    ```shell
    python preprocess.py --src data.txt.char.train.ner --tgt train.conllu
    ```

## Environment
    python3
    torch >= 1.4
    transformers == 3.1.0

## Training 
* Train a simple seqtag model (without CRF)
```shell
python -m supar.cmds.seqtag train -b \
        --train data/aishell_alllabel/train.conllu \
        --dev data/aishell_alllabel/dev.conllu \
        --test data/aishell_alllabel/test.conllu \
        --batch-size 1000 \
        --encoder bert \
        --lr_rate 10 \
        --bert ./bert-base-chinese \
        --use_syntax \
        --mix \
        --synatax_path parser/save/joint-ctb7/ctb7.joint.bert/ \
        -p exp/simple-rhythm-mix-syntax/model \
        -d 4
```
--bert: the name or the path

--lr_rate: the inital learning rate of parameters outside bert equals to lr_rate * 5e-5 

-d: use which gpu

* Train a crf seqtag model
```shell
python -m supar.cmds.crf_seqtag train -b \
        --train data/aishell_alllabel/train.conllu \
        --dev data/aishell_alllabel/dev.conllu \
        --test data/aishell_alllabel/test.conllu \
        --batch-size 1000 \
        --encoder bert \
        --lr_rate 10 \
        --bert ./bert-base-chinese \
        --use_syntax \
        --mix \
        --synatax_path parser/save/joint-ctb7/ctb7.joint.bert/ \
        -p exp/tmp-crf-rhythm-mix-syntax/model \
        -d 3
```

## Evaluate
```shell
python -m supar.cmds.crf_seqtag evaluate --data data/aishell_alllabel/test.conllu \
        -p exp/crf-rhythm-mix-syntax/model \
        -d 3

python -m supar.cmds.seqtag evaluate --data data/aishell_alllabel/test.conllu \
        -p exp/simple-rhythm-mix-syntax/model \
        -d 3
```

## Predict
```shell
python -m supar.cmds.seqtag predict --data data/aishell_alllabel/test.conllu \
        --pred alllabel_test.pred \
        -p exp/simple-rhythm-mix-syntax/model \
        -d 3
python -m supar.cmds.crf_seqtag predict --data data/aishell_alllabel/test.conllu \
        --pred alllabel_test.pred \
        -p exp/crf-rhythm-mix-syntax/model \
        -d 3
```

