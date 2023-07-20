----------------------------------------------

- 코더 : 김경훈
- 리뷰어 : 남희정

----------------------------------------------

PRT(PeerReviewTemplate)

- [ ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [ ] 네.
```python
    model_1 = ChatBot(tokenizer, enc_train, dec_train)
    model_1.train()
    model_1.submit(list(questions[:100]), list(answers[:100]))

    model_2 = ChatBot(tokenizer, enc_train, dec_train, n_layers=6, n_heads=16, d_model=1024, d_ff=4096)
    model_2.train()
    model_2.submit(list(questions[:100]), list(answers[:100]))

    model_3 = ChatBot(tokenizer, enc_train, dec_train, epochs=50)
    model_3.train()
    model_3.submit(list(questions[:100]), list(answers[:100]))

    model_4 = ChatBot(tokenizer, enc_train, dec_train, n_layers=6, n_heads=16, d_model=1024, d_ff=4096, epochs=50)
    model_4.train()
    model_4.submit(list(questions[:100]), list(answers[:100]))
```
이하 생략
상기 코드와 함께 모델의 훈련 결과도 함께 확인하였습니다. 

- [x] 주석을 보고 작성자의 코드가 이해되었나요?
- [ ]  네. 단계별 주석 처리가 이해가기 쉽도록 잘 되었습니다.  
- [x] 코드가 에러를 유발할 가능성이 있나요?
- [ ] 제 코드도 안돌아가서요.. 
- [ ] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [ ] 네네.
- [ ] 코드가 간결한가요?
- [ ] 네네.
 
 ----------------------------------------------

참고 링크 및 코드 개선
