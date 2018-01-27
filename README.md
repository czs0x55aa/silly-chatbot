# silly-chatbot
seq2seq chatbot implement using PyTorch.  
PS:这个版本的代码写的比较凌乱，功能也不完善(没有beam search, antiLM等)。目前没有时间维护，写完硕士论文会考虑更新。
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
This script will create file `dialogue_corpus.txt` in `./data` directory.

### Training Model
```python
python train.py
```
The hyperparameters of model define in configuration file `config.json`.  
In my local environment(GTX1060), training model need about four hours.

### Testing
```python
python chatbot.py
```

#### Test Example
```
> hi .
bot: hi .
> what's your name ?
bot: jacob .
> how are you ?
bot: fine .
> where are you from ?
bot: up north .
> are you happy today ?
bot: yes .
```
The chatbot can answer these simple questions, but in most cases it is a silly bot.

## Reference
- [PyTorch documentation](http://pytorch.org/docs/0.1.12/)
- [seq2seq-translation](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation)
- [tensorflow_chatbot](https://github.com/llSourcell/tensorflow_chatbot)
- [Cornell Movie Dialogs Corpus](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus)
