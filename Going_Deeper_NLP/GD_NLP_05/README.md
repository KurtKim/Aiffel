>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|번역기 모델 학습에 필요한 텍스트 데이터 전처리가 잘 이루어졌다.|데이터 정제, SentencePiece를 활용한 토큰화 및 데이터셋 구축의 과정이 지시대로 진행되었다.|⭐|
>|2|Transformer 번역기 모델이 정상적으로 구동된다.|Transformer 모델의 학습과 추론 과정이 정상적으로 진행되어, 한-영 번역기능이 정상 동작한다.|⭐|
>|3|테스트 결과 의미가 통하는 수준의 번역문이 생성되었다.|제시된 문장에 대한 그럴듯한 영어 번역문이 생성되며, 시각화된 Attention Map으로 결과를 뒷받침한다.|⭐|

----------------------------------------------

- 코더 : 김경훈
- 리뷰어 : 소용현

----------------------------------------------

PRT(PeerReviewTemplate)

- [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
![image](https://github.com/KurtKim/aiffel/assets/100551891/b7450307-efcb-45d0-bf3c-8438efc73b77)

하이퍼 파라미터를 변경해서 학습시키고, 결과를 비교해 보았다.

![image](https://github.com/KurtKim/aiffel/assets/100551891/6ec6f787-7fc2-487e-b583-c6f2f74d2a03)

적절한 시각화가 이루어졌다.

- [o] 주석을 보고 작성자의 코드가 이해되었나요?
![image](https://github.com/KurtKim/aiffel/assets/100551891/b4983e20-7613-4c7d-b307-7223f073f15b)

별도의 설명을 통해 이해하기 쉽게 되어 있다.

- [x] 코드가 에러를 유발할 가능성이 있나요?
```
class Translator():
    def __init__(self, src_tokenizer, tgt_tokenizer, 
                 n_layers=6, d_model=512, n_heads=8, d_ff=2048, pos_len=200, dropout=0.2, shared=True,
                 batch_size=64, epochs=20):
```
pos_len=200 이부분을 최대 seq로 맞춰주면 좋지 않을까요?

- [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
인터뷰 결과 제대로 이해하고 작성하였다.
- [o] 코드가 간결한가요?

```
model_try_1 = Translator(ko_tokenizer, en_tokenizer, n_layers=2, d_model=512, batch_size=64, epochs=20)
model_try_1.train(enc_train, dec_train)
```
별도의 클래스 선언으로 간결한 코드로 테스트 가능하도록 되어 있다.
----------------------------------------------

참고 링크 및 코드 개선
