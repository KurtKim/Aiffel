>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|분류 모델의 accuracy가 기준 이상 높게 나왔는가?|3가지 단어 개수에 대해 8가지 머신러닝 기법을 적용하여 그중 최적의 솔루션을 도출하였다.|⭐|
>|2|분류 모델의 F1 score가 기준 이상 높게 나왔는가?|Vocabulary size에 따른 각 머신러닝 모델의 성능변화 추이를 살피고, 해당 머신러닝 알고리즘의 특성에 근거해 원인을 분석하였다.|⭐|
>|3|딥러닝 모델을 활용해 성능이 비교 및 확인되었는가?|동일한 데이터셋과 전처리 조건으로 딥러닝 모델의 성능과 비교하여 결과에 따른 원인을 분석하였다.|⭐|

----------------------------------------------

- 코더 : 김경훈
- 리뷰어 : 남희정

----------------------------------------------

PRT(PeerReviewTemplate)

- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
*  네. 코드는 정상적으로 동작하고 주어진 문제를 해결하였습니다. 특히 아래 코드를 통해 실험의 결과값들을 비교하기 쉽게 정리하 부분을 분석하며 또 한가지 더 배웠습니다.
``` python
  import pandas as pd

  def concatenate_result(_type):
    result = []
    for num_words in ['5k', '10k', '15k', '20k', 'All']:
        _df = pd.read_csv(f'./results/{_type}_{num_words}.csv')
        _df.index.name = 'num_words'
        _df.index = [f'{num_words}']
        result.append(_df)
    
    return pd.concat(result)
    history_5k, y_5k, pred_5k = d_experiment(num_words=5000)
    def d_experiment(num_words=None):
    
    key = 'All' if num_words is None else f'{str(num_words // 1000) + "k"}'
    ```
    
* 아래 코드와 같이 딥러닝 모델 실험에서도 동일한 단어갯수를 test 할 수 있도록 함수화 하였습니다.   
    ``` python    
    x_train, y_train, x_test, y_test = get_data(num_words=num_words)
    tfidfv_train, tfidfv_test = vectorize_data(x_train, x_test)

    num_classes = max(y_train) + 1

    word_vector_dim = 32  

    model = Sequential()
#         model.add(Embedding(num_words, word_vector_dim))
#         model.add(LSTM(128))

    model.add(Dense(128, activation='relu', input_shape=(tfidfv_train.todense().shape[1],)))  

    model.add(Dense(128, activation='relu')) 
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  

    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    history = model.fit(tfidfv_train.todense(), #.toarray(),
                        y_train,
                        epochs=50,
                        batch_size=64,
                        callbacks=[es],
                        validation_split=0.2,
                        verbose=1)

    predict = model.predict(tfidfv_test.todense()).argmax(axis=1)

    return history, y_test, predict
    ```

- [x] 주석을 보고 작성자의 코드가 이해되었나요?
   *네 잘 이해 되었습니다.
- [x] 코드가 에러를 유발할 가능성이 있나요?
   * 고민을 해 보았지만 아직 제가 더 실력을 쌀아야 할것 같습니다. 
- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
   * 네. 잘 이해 하신것 같습니다. 
- [x] 코드가 간결한가요?
   *네. 간결하게 잘 짜여졌습니니다. 
 
 ----------------------------------------------

참고 링크 및 코드 개선


