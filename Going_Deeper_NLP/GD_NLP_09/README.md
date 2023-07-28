>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|기존 데이터셋을 추가 정제하고, generation 성능을 끌어올리기 위한 기법들을 실험해 모델 perfomance를 향상시켜보았는가?|기존 데이터셋의 문제점을 분석하고 전처리 전략을 수립해 추가 정제를 진행했다. Beam search, Top-k(p) sampling 등 최선의 디코딩 전략을 수립해 향상된 모델 추론 결과를 제시했다. BLEU, ROUGE 등 생성된 텍스트를 평가하기 위한 메트릭을 적용한 정량적인 평가 결과와 주관적인 평가를 비교분석하였다.|⭐|
>|2|새로운 데이터를 수집해 전처리를 수행하여 모델을 재학습시켜보았는가?|모두의 말뭉치, AI hub 등에 공개된 데이터를 사용해 추가 데이터셋을 구축하기 위한 기준과 근거를 수립했다. ChatGPT API나 다양한 한국어 benchmark 데이터셋을 활용해 Human Feedback 을 대체할 수 있는 아이디어를 구현했다. 위를 바탕으로 SFT, RM, PPO 세 단계에 필요한 각 데이터셋을 적절히 구축하여, 모델 추론 결과와 수립한 가설을 비교해보았다.|⭐|
>|3|학습 전략 또는 foundation model을 변경해 모델을 재학습시켜보았는가?|더 적절한 Instruction Tuning 기법을 적용해 SFT를 해보거나, Reward Model의 ranking algorithm을 개선해보았다. KoGPT-2가 아닌 다른 모델을 initial model로 사용하여 모델 학습을 성공시켰다. 허깅페이스의 accelerate, bitsandbytes 라이브러리 등을 사용하여 더 큰 스케일의 모델로 ChatGPT를 re-building해 모델 성능을 향상시켰다.|⭐|


## **Code Peer Review Templete**
------------------
- 코더 : 김경훈
- 리뷰어 : 김동규

## **PRT(PeerReviewTemplate)**
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**
- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**
- [x] **5. 코드가 간결한가요?**

## 증거

### 항목 1
별도로 주입한 데이터 있음

```
train_dataset = SFT_dataset(
    data_path_1_SFT='./data/KoAlpaca_v1.1.jsonl', 
    tokenizer=tokenizer)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

print('input : %s'%train_dataset.input_ids[0])
print('output: %s'%train_dataset.labels[0])
```

개량을 위해 다른 모델 도입
```
model = AutoModelForCausalLM.from_pretrained('EleutherAI/polyglot-ko-1.3b')

tokenizer = AutoTokenizer.from_pretrained(
    'EleutherAI/polyglot-ko-1.3b', 
    bos_token='</s>', eos_token='</s>', unk_token='</s>', pad_token='</s>',
    padding_side="right", model_max_length=512,
)
```
### 항목 2
주석으로 깔끔하게 정리함함
```
from typing import Optional, Dict, Sequence

class SFT_dataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, verbose=False):
        super(SFT_dataset, self).__init__()
        logging.warning("Loading data...")
        
        sources = []
        targets = []
        
        ###############################################
        
        data_path_1_SFT = './data/kochatgpt_1_SFT.jsonl'
        with open(data_path_1_SFT, "r", encoding='utf-8-sig') as json_file:
            list_data_dict_1 = json.load(json_file)

        PROMPT_DICT_1 = {
            "prompt_input": (
                "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
            )
        }

        prompt_input_1 = PROMPT_DICT_1["prompt_input"]

        for example in list_data_dict_1:
            tmp = prompt_input_1.format_map(example)
            sources.append(tmp)

        for example in list_data_dict_1:
            targets.append(f"{example['completion']}{tokenizer.eos_token}")
        
        ###############################################
        
        data_path_2_SFT = './data/KoAlpaca_v1.1.jsonl'
        list_data_dict_2 = []
        with open(data_path_2_SFT, "r", encoding='utf-8-sig') as json_file:
            for line in json_file:
                list_data_dict_2.append(json.loads(line))
                
        PROMPT_DICT_2 = {
            "prompt_input": (
                "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
            )
        }

        prompt_input_2 = PROMPT_DICT_2["prompt_input"]
        
        for example in list_data_dict_2:
            tmp = prompt_input_2.format_map(example)
            sources.append(tmp)

        for example in list_data_dict_2:
            targets.append(f"{example['output']}{tokenizer.eos_token}")
        
        ###############################################
        
        ...

```

### 항목 3

파일 경로를 올바르게 설정하여 오류를 방지함

```
train_dataset = SFT_dataset(
    data_path_1_SFT='./data/KoAlpaca_v1.1.jsonl', 
    tokenizer=tokenizer)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

print('input : %s'%train_dataset.input_ids[0])
print('output: %s'%train_dataset.labels[0])
```

### 항목 4

생성을 위해 적절한 파이프라인을 선택하고 활용했다.
```
generator = pipeline('text-generation', model='./models', tokenizer=tokenizer)

generation_args = dict(   
    num_beams=4,
    repetition_penalty=2.0,
    no_repeat_ngram_size=4,
    eos_token_id=375, # \n   
    max_new_tokens=64,
    do_sample=True,
    top_k=50,
    early_stopping=True
)

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
    )
}

list_prompt = ['불고기용 고기 한우에요?',
               '리처드 닉슨이 43대 부통령직을 수행한 년도는?',
               '시카고 오헤어 국제공항은 어디에 있어?',
               '오늘 미세먼지 어때?']

list_prompt = [PROMPT_DICT['prompt_input'].format_map({'prompt' : tmp}) for tmp in list_prompt]

list_result = generator(list_prompt, **generation_args)   
for prompt, result in zip(list_prompt, list_result):
    print()
    print((result[0]['generated_text']))
```

### 항목 5

트레이너 객체를 사용하여 코드가 더 간단 간결해짐

```
training_args = TrainingArguments(
    output_dir="./logs",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    warmup_steps=5,
    prediction_loss_only=True,
    fp16 = True,
    optim="adafactor",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)
```


## **참고링크 및 코드 개선 여부**
