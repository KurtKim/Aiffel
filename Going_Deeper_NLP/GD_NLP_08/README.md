>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|모델과 데이터를 정상적으로 불러오고, 작동하는 것을 확인하였다.|klue/bert-base를 NSMC 데이터셋으로 fine-tuning 하여, 모델이 정상적으로 작동하는 것을 확인하였다.|⭐|
>|2|Preprocessing을 개선하고, fine-tuning을 통해 모델의 성능을 개선시켰다.|Validation accuracy를 90% 이상으로 개선하였다.|⭐|
>|3|모델 학습에 Bucketing을 성공적으로 적용하고, 그 결과를 비교분석하였다.|Bucketing task을 수행하여 fine-tuning 시 연산 속도와 모델 성능 간의 trade-off 관계가 발생하는지 여부를 확인하고, 분석한 결과를 제시하였다.|⭐|

----------------------------------------------

- 코더 : 김경훈
- 리뷰어 : 소용현

----------------------------------------------

PRT(PeerReviewTemplate)

- [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
![image](https://github.com/KurtKim/aiffel/assets/100551891/b54f0a6d-a5ca-4065-ad94-23f9ffa9bccb)
90%이상 정확도 달성하였습니다.
- [x] 주석을 보고 작성자의 코드가 이해되었나요?
주석이 없습니다 ㅠㅠ
- [x] 코드가 에러를 유발할 가능성이 있나요?
없습니다.
- [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
네 이해하고 작성하였습니다.
- [o] 코드가 간결한가요?
```
class Classifier():
    def __init__(self, model_name, dataset, training_arguments):
        super(Classifier, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        self.dataset = dataset
        self.trainer = self._set(training_arguments)
    
    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return { 'accuracy': accuracy_score(labels, preds) }
    
    
    def _set(self, training_arguments):
        return Trainer(
            model=self.model,           
            args=training_arguments,           
            train_dataset=self.dataset.train,
            eval_dataset=self.dataset.valid,       
            compute_metrics=self.compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
    
    
    def fine_tuning(self):
        return self.trainer.train()
        
    
    def evalutate(self):
        return self.trainer.evaluate(self.dataset.test)
```
학습에 필요한 클래스를 정의하여 코드의 중복을 줄였습니다.

참고 링크 및 코드 개선
