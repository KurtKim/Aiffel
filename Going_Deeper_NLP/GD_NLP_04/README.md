>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|번역기 모델 학습에 필요한 텍스트 데이터 전처리가 한국어 포함하여 잘 이루어졌다.|구두점, 대소문자, 띄어쓰기, 한글 형태소분석 등 번역기 모델에 요구되는 전처리가 정상적으로 진행되었다.|⭐|
>|2|Attentional Seq2seq 모델이 정상적으로 구동된다.|seq2seq 모델 훈련 과정에서 training loss가 안정적으로 떨어지면서 학습이 진행됨이 확인되었다.|⭐|
>|3|테스트 결과 의미가 통하는 수준의 번역문이 생성되었다.|테스트용 디코더 모델이 정상적으로 만들어져서, 정답과 어느 정도 유사한 영어 번역이 진행됨을 확인하였다.|⭐|

----------------------------------------------

- 코더 : 김경훈
- 리뷰어 : Donggyu Kim

----------------------------------------------

## PRT(PeerReviewTemplate)
- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
- [x] 주석을 보고 작성자의 코드가 이해되었나요?
- [x] 코드가 에러를 유발할 가능성이 있나요?
- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [x] 코드가 간결한가요? 

## The reasons of marks

### Item 1

#### 1st mission
It was nicely done.

Data preprocessor
```python
import re

def preprocess_sentence(sentence, s_token=False, e_token=False):
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!가-힣ㄱ-ㅎㅏ-ㅣ]+", " ", sentence)
    sentence = sentence.strip()

    if s_token:
        sentence = '<start> ' + sentence

    if e_token:
        sentence += ' <end>'
    
    return sentence
```

Data tokenizer
```python
from konlpy.tag import Mecab
import tensorflow as tf

def tokenize(corpus, is_kor=True):
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    
    if is_kor:
        mecab = Mecab()
        corpus = [' '.join(mecab.morphs(sen)) for sen in corpus]
        
    tokenizer.fit_on_texts(corpus)
    tensor = tokenizer.texts_to_sequences(corpus)
    
    if is_kor:
        tensor = [list(reversed(sen)) for sen in tensor]
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='pre', maxlen=40)
    else:
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=40)
        
    return tensor, tokenizer
```

He check the data too.

```
print("Korean:", kor_corpus[100])   
print("English:", eng_corpus[100])  
>> Korean: 다우는 현재 포인트 그리고 나스닥은 포인트 떨어졌습니다 .
>> English: <start> the dow currently down points nasdaq off . <end>

print("Korean:", kor_corpus[150])   
print("English:", eng_corpus[150])  
>> Korean: 인권 침해
>> English: <start> all rights reserved . <end>

print("Korean:", kor_corpus[200])   
print("English:", eng_corpus[200])  
>> Korean: 고령화로 경제 성장률 침체된다
>> English: <start> aging population results in the stagnant economic growth . <end>
```

#### mission 2

The model is very clear and the code is also nice.
This code is worth learning.
Both the class structure and the interface are perfect.
I also thought it could be rude for him to comment on the code.

```
class TranslateModel():
    def __init__(self, enc_tokenizer, dec_tokenizer, batch_size=64, units=1024, embedding_dim=512, epochs=10):
        super(TranslateModel, self).__init__()
        
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        
        self.BATCH_SIZE = batch_size
        self.units = units
        self.embedding_dim = embedding_dim
        
        SRC_VOCAB_SIZE = len(enc_tokenizer.index_word) + 1
        TGT_VOCAB_SIZE = len(dec_tokenizer.index_word) + 1
            
        self.encoder = Encoder(SRC_VOCAB_SIZE, embedding_dim, units)
        self.decoder = Decoder(TGT_VOCAB_SIZE, embedding_dim, units)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        
        self.EPOCHS = epochs
        
        self.enc_shape = None
        self.dec_shape = None
                  
            
    def summary(self):
        
        sequence_len = 40

        sample_enc = tf.random.uniform((self.BATCH_SIZE, sequence_len))
        sample_output = self.encoder(sample_enc)

        print ('Encoder Output:', sample_output.shape)

        sample_state = tf.random.uniform((self.BATCH_SIZE, self.units))

        sample_logits, h_dec, attn = self.decoder(
            tf.random.uniform((self.BATCH_SIZE, 1)), 
            sample_state, 
            sample_output)

        print ('Decoder Output:', sample_logits.shape)
        print ('Decoder Hidden State:', h_dec.shape)
        print ('Attention:', attn.shape)
        
    
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_mean(loss)
    
    
    @tf.function
    def train_step(self, src, tgt):
        bsz = src.shape[0]
        loss = 0

        with tf.GradientTape() as tape:
            enc_out = self.encoder(src)
            h_dec = enc_out[:, -1]

            dec_src = tf.expand_dims([self.dec_tokenizer.word_index['<start>']] * bsz, 1)

            for t in range(1, tgt.shape[1]):
                pred, h_dec, _ = self.decoder(dec_src, h_dec, enc_out)

                loss += self.loss_function(tgt[:, t], pred)
                dec_src = tf.expand_dims(tgt[:, t], 1)

        batch_loss = (loss / int(tgt.shape[1]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss
    
    
    @tf.function
    def eval_step(self, src, tgt):
        ...
    
    
    def train(self, enc_train, enc_test, dec_train, dec_test):
        ...
    ...
```

#### mission 3
Unfortunately, his model's train result was not perfact.
But the code is accepable.

```
model.translate('말하기 전에 생각했나요?')

Input: 말하기 전에 생각했나요 ?
Predicted translation: obama ups <end> 
```

### Item 2
Although he doesn't write comments in the code, I can understood his code.
Because his code is clear to understand easily. 
Readability is one of important things in the zen of python.
And he explained the code using markdown.

```
## Step 3. 데이터 토큰화
앞서 정의한 tokenize()함수를 사용해 데이터를 텐서로 변환하고 각각의 tokenizer를 얻으세요! 단어의 수는 실험을 통해 적당한 값을 맞춰주도록 합니다! (최소 10,000 이상!)

❗ 주의: 난이도에 비해 데이터가 많지 않아 훈련 데이터와 검증 데이터를 따로 나누지는 않습니다.

한글 토큰화는 KoNLPy의 mecab클래스를 사용합니다.
```

```
한국어를 영어로 잘 번역해 줄 멋진 Attention 기반 Seq2seq 모델을 설계하세요! 앞서 만든 모델에 Dropout모듈을 추가하면 성능이 더 좋아집니다! Embedding Size와 Hidden Size는 실험을 통해 적당한 값을 맞춰 주도록 합니다!
```

### Item 3

the `import` was close on the usage.
It can be blocked the Import erorrs.

```python
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import re
import numpy as np


class TranslateModel():
    def __init__(self, enc_tokenizer, dec_tokenizer, batch_size=64, units=1024, embedding_dim=512, epochs=10):
        super(TranslateModel, self).__init__()
        
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
       ...
```

### Item 4

Yes, he used some code not in the study materials.
It can be seen that he understand more things at least rather that me.
He organised the code in the model class as well.

```
class TranslateModel():
    def __init__(self, enc_tokenizer, dec_tokenizer, batch_size=64, units=1024, embedding_dim=512, epochs=10):
        ...       
            
    def summary(self):
        
        sequence_len = 40

        sample_enc = tf.random.uniform((self.BATCH_SIZE, sequence_len))
        sample_output = self.encoder(sample_enc)

        print ('Encoder Output:', sample_output.shape)

        sample_state = tf.random.uniform((self.BATCH_SIZE, self.units))

        sample_logits, h_dec, attn = self.decoder(
            tf.random.uniform((self.BATCH_SIZE, 1)), 
            sample_state, 
            sample_output)

        print ('Decoder Output:', sample_logits.shape)
        print ('Decoder Hidden State:', h_dec.shape)
        print ('Attention:', attn.shape)
        
    
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_mean(loss)
    
    
    @tf.function
    def train_step(self, src, tgt):
        bsz = src.shape[0]
        loss = 0

        with tf.GradientTape() as tape:
            enc_out = self.encoder(src)
            h_dec = enc_out[:, -1]

            dec_src = tf.expand_dims([self.dec_tokenizer.word_index['<start>']] * bsz, 1)

            for t in range(1, tgt.shape[1]):
                pred, h_dec, _ = self.decoder(dec_src, h_dec, enc_out)

                loss += self.loss_function(tgt[:, t], pred)
                dec_src = tf.expand_dims(tgt[:, t], 1)

        batch_loss = (loss / int(tgt.shape[1]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss
    
    
    @tf.function
    def eval_step(self, src, tgt):
        bsz = src.shape[0]
        loss = 0

        enc_out = self.encoder(src)

        h_dec = enc_out[:, -1]

        dec_src = tf.expand_dims([self.dec_tokenizer.word_index['<start>']] * bsz, 1)

        for t in range(1, tgt.shape[1]):
            pred, h_dec, _ = self.decoder(dec_src, h_dec, enc_out)

            loss += self.loss_function(tgt[:, t], pred)
            dec_src = tf.expand_dims(tgt[:, t], 1)

        batch_loss = (loss / int(tgt.shape[1]))

        return batch_loss
```

### Item 5

By using OOP, his code is very simple but easily understandable.

The usage was also simple.

**model train**
```
model.train(kor_train, kor_test, eng_train, eng_test)

```

**model inference**
```python
model.translate('무슨 영화 좋아하세요? 부귀영화요')
```

**visualize**
```python
def translate(self, sentence):
    result, sentence, attention = self.evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention = attention[:len(result.split()), :len(sentence.split())]
    self.plot_attention(attention, sentence.split(), result.split(' ')) <--- this one
```


 ----------------------------------------------

참고 링크 및 코드 개선
No improvements I would comment on.


