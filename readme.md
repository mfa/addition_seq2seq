## Addition in PyTorch

This example is inspired by https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py


This code requires Python 3.7!!
(Because it uses dataclasses)


### Install

```
pip install -r code/requirements.txt
```


### Generate Data

```
python code/gendata.py -o data/train_100000_1000.csv -s 100000 --high 1000
python code/gendata.py -o data/val_100_10.csv -s 100 --high 10
python code/gendata.py -o data/val_100_100.csv -s 100 --high 100
python code/gendata.py -o data/val_100_1000.csv -s 100 --high 1000
cat data/val_100_10.csv data/val_100_100.csv data/val_100_1000.csv > data/val_300.csv
```

### Train

```
python code/main.py \
  --train-data-path=data/train_add.csv \
  --val-data-path=data/val_add.csv \
  --attn_method=dot \
  --batch-size=1024 \
  --epochs=100
```

#### train on floydhub

```
floyd run --task train1000
```

### tensorboard

(locally)

```
docker run -p 0.0.0.0:6006:6006 -it -v `pwd`/logs:/root/logs  tensorflow/tensorflow tensorboard --logdir /root/logs/
```
