>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|다양한 방법으로 Text Classification 태스크를 성공적으로 구현하였다.|3가지 이상의 모델이 성공적으로 시도됨|⭐|
>|2|gensim을 활용하여 자체학습된 혹은 사전학습된 임베딩 레이어를 분석하였다.|gensim의 유사단어 찾기를 활용하여 자체학습한 임베딩과 사전학습 임베딩을 비교 분석함|⭐|
>|3|한국어 Word2Vec을 활용하여 가시적인 성능향상을 달성했다.|네이버 영화리뷰 데이터 감성분석 정확도를 85% 이상 달성함|⭐|

----------------------------------------------

- 코더 : 김경훈
- 리뷰어 : 남희정

----------------------------------------------

PRT(PeerReviewTemplate)

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- 네. 
- 데이터의 중복 제거<br>
NaN 결측치 제거<br>
한국어 토크나이저로 토큰화<br>
불용어(Stopwords) 제거<br>
사전word_to_index 구성<br>
텍스트 스트링을 사전 인덱스 스트링으로 변환<br>
X_train, y_train, X_test, y_test, word_to_index 리턴<br>
데이터셋 내 문장 길이 분포<br>
적절한 최대 문장 길이 지정<br>
keras.preprocessing.sequence.pad_sequences 을 활용한 패딩 추가<br>
이후 데이터 분석과정, Validation, 모델훈련, 시각화등 주어진 미션이 잘 수행 되었습니다. 
특히 예시로 주어졌던 load_data(train_data, test_data, num_words=10000): 함수내 필요한 값 setting이 잘 적용되었습니다.<br>

<code> words = np.concatenate(X_train).tolist()
 counter = Counter(words)
 counter = counter.most_common(10000-4)
 vocab = ['<PAD>', '<BOS>', '<UNK>', '<UNUSED>'] + [key for key, _ in counter]
 word_to_index = {word:index for index, word in enumerate(vocab)}
    
 def wordlist_to_indexlist(wordlist):
     return [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in wordlist]</code><br>
5종류의 Sequential 모델 test을 사용해 다양한 학습을 진행하였습니다.<br>
- [X] 주석을 보고 작성자의 코드가 이해되었나요?<br>
  주석과 markdown을 통해 코드를 이해하기 쉽도록 잘 설명하였습니다. 
- [X] 코드가 에러를 유발할 가능성이 있나요?<br>
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)<br>
  네. <br>
- [X] 코드가 간결한가요?<br>
  네. 간결하게 잘 정리 되었습니다. 

----------------------------------------------

참고 링크 및 코드 개선<br>
  코드를 개선을 제안하기엔 제가 부족한것 같고, 오히려 작성하신 코드를 보면서 제 코드에 대한 개선점을 생각해 보았습니다.


