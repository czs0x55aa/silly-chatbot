# silly-chatbot
Work in progress!

## Requirements
- Python 2.7
- Pytorch 0.12

## Corpus
- [Cornell Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## Usage
### Preprocess
```python
python preprocess.py
```
this script will create file `dialogue_corpus.txt` in `./data` directory.

### Training Model
```python
python train.py
```
the hyperparameters of model defined in configuration file `config.json`.

### Test bot
```python
python chatbot
```

## Reference
- [seq2seq-translation](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation)
- [tensorflow_chatbot](https://github.com/llSourcell/tensorflow_chatbot)
- [Cornell Movie Dialogs Corpus](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus)
- [PyTorch documentation](http://pytorch.org/docs/0.1.12/)
