>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|한글 코퍼스를 가공하여 BERT pretrain용 데이터셋을 잘 생성하였다.|MLM, NSP task의 특징이 잘 반영된 pretrain용 데이터셋 생성과정이 체계적으로 진행되었다.|⭐|
>|2|구현한 BERT 모델의 학습이 안정적으로 진행됨을 확인하였다.|학습진행 과정 중에 MLM, NSP loss의 안정적인 감소가 확인되었다.|⭐|
>|3|1M짜리 mini BERT 모델의 제작과 학습이 정상적으로 진행되었다.|학습된 모델 및 학습과정의 시각화 내역이 제출되었다.|⭐|

----------------------------------------------

- 코더 : 김경훈
- 리뷰어 : 소용현

----------------------------------------------

PRT(PeerReviewTemplate)

- [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  ![image](https://github.com/KurtKim/aiffel/assets/100551891/6e476871-f176-4224-8229-776333e814f5)
  주어진 문제를 해결하였습니다.
- [o] 주석을 보고 작성자의 코드가 이해되었나요?
```
def load_pre_train_data(vocab, filename, n_seq, count=None):
    """
    학습에 필요한 데이터를 로드
    :param vocab: vocab
    :param filename: 전처리된 json 파일
    :param n_seq: 시퀀스 길이 (number of sequence)
    :param count: 데이터 수 제한 (None이면 전체)
    :return enc_tokens: encoder inputs
    :return segments: segment inputs
    :return labels_nsp: nsp labels
    :return labels_mlm: mlm labels
    """
```
함수의 인풋 아웃풋이 잘 정리되어 있습니다.
- [x] 코드가 에러를 유발할 가능성이 있나요?
- [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
  인터뷰 결과 제대로 이해하고 작성하였습니다.
- [o] 코드가 간결한가요?
간결하게 작성되어 있습니다.
```
def lm_loss(y_true, y_pred):
    """
    loss 계산 함수
    :param y_true: 정답 (bs, n_seq)
    :param y_pred: 예측 값 (bs, n_seq, n_vocab)
    """
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)

    mask = tf.cast(tf.math.not_equal(y_true, 0), dtype=loss.dtype)
    loss *= mask
    return loss * 20
```

 ----------------------------------------------

참고 링크 및 코드 개선
