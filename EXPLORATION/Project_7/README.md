>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|pix2pix 모델 학습을 위해 필요한 데이터셋을 적절히 구축하였다.|데이터 분석 과정 및 한 가지 이상의 augmentation을 포함한 데이터셋 구축 과정이 체계적으로 제시되었다.|⭐|
>|2|pix2pix 모델을 구현하여 성공적으로 학습 과정을 진행하였다.|U-Net generator, discriminator 모델 구현이 완료되어 train_step의 output을 확인하고 개선하였다.|⭐|
>|3|학습 과정 및 테스트에 대한 시각화 결과를 제출하였다.|10 epoch 이상의 학습을 진행한 후 최종 테스트 결과에서 진행한 epoch 수에 걸맞은 정도의 품질을 확인하였다.|⭐|

----------------------------------------------

- 코더 : 김경훈
- 리뷰어 : 이효준

----------------------------------------------

PRT(PeerReviewTemplate)

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
> 마크다운으로 순서있게 정리되어 있어 쉽게 이해되며 읽을 수 있었습니다.
- [X] 코드가 에러를 유발할 가능성이 있나요?
> 에러는 아니지만 __작성 의도와 다른 동작을 보이는__ 코드가 있습니다.  
> `Class Discriminator(Model)`에서 `첫번째(i==0) Discblock`과 `다섯번째(i==4) Discblock`에 `Batch Normalization`은 `False`을 적고 나머지 에서는 `True` 결과를 얻으려 했던 것으로 보입니다.  
> 이 부분만 보면 아래와 같이 동작하게 됩니다.
```python
# 작성하신 코드
for i, f in enumerate([64, 128, 256, 512, 1]):
    print(f"{i}번째 결과: {False if i ==0 and i == 4 else True}")
```
> 작성하신 코드 실행 결과  
> 0번째 결과: True  
> 1번째 결과: True  
> 2번째 결과: True  
> 3번째 결과: True  
> 4번째 결과: True  
```python
# 개선 방안
for i, f in enumerate([64, 128, 256, 512, 1])):
    print(f"{i}번째 결과: {True if i !=0 and i !=4 else False}")
```
> 개선 방안 코드 실행 결과  
> 0번째 결과: False  
> 1번째 결과: True  
> 2번째 결과: True  
> 3번째 결과: True  
> 4번째 결과: False

- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
> 어려운 부분없이 잘 이해하고 작성하였습니다.

- [X] 코드가 간결한가요?
> pyplot을 for문으로 간결하게 시각화 해주는 부분이 인상 깊었습니다.
```python
test_count = len(os.listdir(test_path))
for index, file in enumerate(os.listdir(test_path)):
    sketch, colored = load_img(f'{test_path}/{file}')
    
    pred = generator(tf.expand_dims(sketch, 0))
    pred = denormalize(pred)
    
    plt.subplot(3, test_count, index + 1)
    plt.imshow(denormalize(sketch))
    plt.title(f'Input Image - {index + 1}')
    
    plt.subplot(3, test_count, index + 1 + test_count)
    plt.imshow(denormalize(colored))
    plt.title(f'Ground Truth -  {index + 1}')
    
    plt.subplot(3, test_count, index + 1 + (test_count * 2))
    plt.imshow(pred[0])
    plt.title(f'Predicted Image -  {index + 1}')
```
----------------------------------------------

참고 링크 및 코드 개선
> 개선이 필요하다고 생각되는 부분을 위에 작성 하였습니다.
