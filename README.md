







# 





# 주제 분류 프로젝트



모델 구조의 변경 없이 Data-Centric 관점으로 텍스트의 주제를 분류하는 태스크입니다.

#부스트캠프5기 #자연어처리

종료| 2023.05.24 ~ 2023.06.01 19:00



## 개요

우리는 살아가면서 다양한 자연어 문장들을 마주하게 됩니다. 초등학교 때 쓰던 알림장에서부터, 시험 공부를 위해 들여다본 책이나, 성인이 된 후에도 계속해서 정보를 얻기 위한 글이나, 영상의 자막 모두 자연어 문장들로 이루어져 있습니다. 하다 못해 지인들과 주고 받는 메세지와 편지들, 업무 전달을 위한 메신저와 문서들도 모두 자연어 문장들로 이루어져 있습니다. 어렸을 때부터 우리는 무의식적으로 각 자연어 문장들이 어떤 주제로 이루어져 있는지 판단 후 내용을 파악하게 됩니다.

![](https://lh6.googleusercontent.com/08NseNRg95yqSQMIFbtaBtXzu1GlEfonO00pFmKb2g4OrU6cdvtzcFV71pe9c1w7x7lNrvpXANhWHYaN1edVkv-oyIHEKcU4aMP00-iD5C97SUtiV2rQ-98svYmxtam4wKBPV-ilmCQuAdpqTBM6xvY)

그렇다면 사람이 아니라 딥러닝 모델은 어떨까요?

자연어를 독해 및 분석 과정을 거쳐 주어진 태스크를 수행하기 위해서는 자연어의 주제에 대한 이해가 필수적입니다. Topic Classification task는 모델이 자연어를 잘 이해하고 있는지 평가할 수 있는 가장 간단한 task입니다.

그 중에서도 KLUE-Topic Classification benchmark는 뉴스의 헤드라인을 통해 그 뉴스가 어떤 topic을 갖는지를 분류해내는 task입니다. 각 자연어 데이터에는 생활문화(Society), 스포츠(Sports), 세계(World), 정치(Politics), 경제(Economy), IT과학(IT/Science), 사회(Society) 등 다양한 주제 중 하나가 라벨링 되어 있습니다.

### 데이터

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

- input : 약 9100개의 뉴스 헤드라인과 url, 작성 날짜

- output : 각 뉴스 헤드라인의 주제 (생활문화, 스포츠, 세계, 정치, 경제, IT과학, 사회 중 하나)

### 룰

Data-Centric 의 취지에 맞게, 베이스라인 모델의 수정 없이 오로지 데이터의 수정으로만 성능 향상을 이끌어내야 합니다. 베이스라인 코드의 수정이 없는 선에서, 모든 방법을 적용할 수 있습니다. 대회 종료 후 베이스라인 코드의 변경이 확인되는 경우, 리더보드에서 제외됩니다.

[가능한 방법]

- Generation Model (GPT3, T5, ChatGPT, GPT-4 등)을 통한 Synthetic Data 생성

- 각종 Data Augmentation 기법 적용

- Data sampling 기법

- negative sampling 등

- Data Quality Control

- Data labeling error detection, Data Cleaning, Data Filtering등

[불가능한 방법]

- 베이스라인 코드의 변경이 수행되는 모든 방법들

- Active learning, Curriculum learning등

(사진 출처: [https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a](https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a))

본 대회는 KLUE Topic classification의 공식 리더보드(https://klue-benchmark.com/tasks/66/leaderboard/task)와 동일한 평가 방법을 사용합니다. 평가 지표(Evaluation metrics)는 F1 점수입니다. Public data와 Private data는 dev dataset에서 무작위로 50:50으로 선정되며, 각각을 통해 Puplic F1과 Private F1을 평가합니다.

Data-Centric NLP 강의의 취지에 맞게 데이터의 변경만 가능합니다. 이에 따라, 대회 종료 후 베이스라인 코드의 변경 여부를 검사하게 됩니다. 만약 코드 변경이 확인되는 경우 리더보드에서 제외됩니다.

### 세부일정

- 프로젝트 전체 기간 (2주) : 5월 22일 (월) 10:00 ~ 6월 1일 (목) 19:00

- 팀 병합 기간 : 5월 23일 (화) 16:00 까지

- 팀명 컨벤션 : 도메인_팀번호(2자리)조 / ex) NLP_02조, NLP_11조

- 리더보드 제출 오픈 : 5월 24일 (수) 10:00

- 리더보드 제출 마감 : 6월 1일 (목) 19:00

- 최종 리더보드 (Private) 공개 : 6월 1일 (목) 20:00

- GPU 서버 할당 : 5월 22일 (월) 10:00

- GPU 서버 회수 : 6월 2일 (금) 16:00

### 대회 세부 룰

- [대회 참여 제한] NLP 도메인을 수강하고 있는 캠퍼에 한하여 리더보드 제출이 가능합니다.

- [팀 결성 기간] 팀 결성은 대회 페이지 공개 후 2일차 오후 4시까지 필수로 진행해 주세요. 팀이 완전히 결성되기 전까지는 리더보드 제출이 불가합니다.

- [일일 제출횟수] 일일 제출횟수는 '팀 단위 10회'로 제한합니다. (일일횟수 초기화 자정 진행)

- [외부 데이터셋 규정] 모든 외부 데이터셋 사용을 허용하나 해당 데이터셋은 public 하게 공개된 상태이며, 저작권 문제가 없고, 공평하게 추가 비용 없이 접근 가능해야 합니다.

- [기학습 가중치 사용] 베이스라인코드 기학습 가중치 외에는 사용이 불가합니다.

- [코드 작성/변경에 대한 규칙] 데이터를 구성하고 활용하는 방법에 집중해보는 것을 장려하는 취지에서, 제공되는 베이스 코드 중 모델과 관련한 부분을 변경하는 것이 금지되어 있습니다. 대회 종료 후 모델 코드를 검토 후, 베이스 코드의 변경이 확인되는 경우 리더보드에서 제외됩니다.

- [가능한 방법]

- Generation Model (GPT3, T5, ChatGPT, GPT-4 등)을 통한 Synthetic Data 생성

- 각종 Data Augmentation 기법 적용

- Data sampling 기법

- negative sampling 등

- Data Quality Control

- Data labeling error detection, Data Cleaning, Data Filtering등

- [불가능한 방법]

- 베이스라인 코드의 변경이 수행되는 모든 방법들

- Active learning, Curriculum learning등

- [데이터셋 저작권] 대회 데이터셋은 '캠프 교육용 라이선스' 아래 사용 가능합니다. 저작권 관련 세부 내용은 부스트코스 공지사항을 반드시 참고 해주세요.

AI Stages 대회 공통사항

- [Private Sharing 금지] 비공개적으로 다른 팀과 코드 혹은 데이터를 공유하는 것은 허용하지 않습니다.코드 공유는 반드시 대회 게시판을 통해 공개적으로 진행되어야 합니다.

- [최종 결과 검증 절차] 리더보드 상위권 대상으로추후 코드 검수가 필요한 대상으로 판단될 경우 개별 연락을 통해 추가 검수 절차를 안내드릴 수 있습니다. 반드시 결과가 재현될 수 있도록 최종 코드를 정리 부탁드립니다. 부정행위가 의심될 경우에는 결과 재현을 요구할 수 있으며, 재현이 어려울 경우 리더보드 순위표에서 제외될 수 있습니다.

- [공유 문화] 공개적으로 토론 게시판을 통해 모델링에 대한 아이디어 혹은 작성한 코드를 공유하실 것을 권장 드립니다. 공유 문화를 통해서 더욱 뛰어난 모델을 대회 참가자 분들과 같이 개발해 보시길 바랍니다.

- [대회 참가 기본 매너] 좋은 대회 문화 정착을 위해 아래 명시된 행위는 지양합니다.

- 대회 종료를 앞두고 (3일 전) 높은 점수를 얻을 수 있는 전체 코드를 공유하는 행위

- 타 참가자와 토론이 아닌 단순 솔루션을 캐내는 행위

## 데이터 상세

대회에서 사용되는 데이터셋은 KLUE 공식 사이트에서 제공하는 KLUE-TC(YNAT) 데이터셋과 같은 포맷을 가집니다. 제공되는 총 학습 데이터는 45,678개(traintest split을 거친 후의 train data는 31,974개, evaluation dataset는 13,704개)이며, 테스트 데이터는 9,107개 입니다.

기존 KLUE-YNAT 학습 데이터셋에 noise가 섞인 데이터가 일부 섞여있습니다.

![](https://lh4.googleusercontent.com/MV2teeF271tqvJrDcrA-Sa3xWax-LAaz7V9lXHAJyBFqmoOQSttZJUxM5SU2sLfvmALY9h2mfvx95rUEfJcih9FqN-HR_JWepsKO_oKHGaJjjBCB2sLdv7Ic4wAtfo67Fmjytrp6dv5gKe9y_AkakkE)

노이즈 데이터의 경우 전체 학습 데이터의 15%(총 6,852개)에 해당합니다. noise data 중 80%(5,481개)는 G2P를 이용한 text perturbation으로 생성되었으며, prescriptive pronunciation과 descriptive pronunciation을 1:1 비율로 사용하여 noise를 생성하였습니다. 나머지 20%(1,371개)는 labeling 오류로 만들어진 데이터 입니다.

데이터는 아래와 같이 csv 파일로 제공되며, 각 행이 하나의 데이터 샘플입니다. 최대한 깔끔하고 명확한 대회 진행을 위해, KLUE-YNAT 데이터 중 일부 feature만을 사용합니다.

- ID 각 데이터 샘플의 고유번호 입니다.

- text 분류의 대상이 되는 자연어 텍스트입니다. 연합 뉴스 기사의 헤드라인이며, 한국어 텍스트에 일부 영어, 한자 등의 단어가 포함되어 있습니다.

- target 정수로 인코딩 된 라벨입니다. 각 target의 자연어 의미와 각각의 개수는 아래 표와 같습니다.  
  
  ![](https://lh6.googleusercontent.com/kT_xRibRdKTgf_06kGrnRUftS6bxACLuaSUl_mVMYA5h6WGeUgInPZy02J9vdUvRSs67-9A72RHFldJok4avusETI8q-C3j60z5LCno5AH9d6VfSBJIYvtbXAdxLj5bOVgXHXn-qW5OAdSY9AQoo_h0)

- url 해당 데이터 샘플의 뉴스 url 입니다.

- date 해당 데이터 샘플의 뉴스가 작성된 날짜와 시간입니다.

- 평가 데이터셋은 KLUE 공식 사이트에서 제공하는 KLUE-TC(YNAT)의 dev 데이터셋과 동일합니다. 데이터의 포맷은 학습 데이터와 동일하지만, 그 중 target 은 포함되지 않습니다.

- 학습 데이터의 input_text와 평가 데이터의 input_text의 길이 분포 차이는 다음과 같습니다.

![](https://lh3.googleusercontent.com/TsGatT6Nq1tgoe3_OC2v1WnIICKz5DZnJ_2kC0ZOkLFb_y0O7VgZUd1XibSpBn52KL5XNG7HL833KozQmfw_y2u9x4H-B5RO2FCNQcmhyhpeg7fJ-yvzhkC4bthoUPKV_kn7efD-QV3hQjKIhLIrfDQ)

- 총 9,107개의 데이터 중에서 Public, Private 데이터를 각각 50% 비율로 무작위로 선정하였습니다.

- Public (대회 진행중)

- test_data.csv로 만든 submission.csv 파일을 통해 자동으로 public과 관련된 샘플들을 평가하게 됩니다.

- Private (대회 종료후)

- test_data.csv로 만든 submission.csv 파일을 통해 자동으로 private과 관련된 샘플들을 평가하게 됩니다.

주어진 code.tar.gz 파일의 압축을 해제하면 다음과 같은 구조로 파일이 존재합니다

├─code

│  │  baseline.ipynb

│  │  requirements.txt

- baseline.ipynb : 베이스라인 코드입니다. 크게 7가지로 분류됩니다.
1. Load Libraries  
   : 베이스라인 코드 실행에 필요한 모든 라이브러리를 로드합니다.

2. Set Hyperparameters  
   : cuda 사용을 위한 device 설정과 경로 지정, 시드 설정을 수행합니다. 또한 max sequence length, batch size 등의 hyperparameter도 지정합니다.

3. Load Tokenizer and Model  
   : 사전 학습된 tokenizer와 model을 로드합니다. 본 베이스라인 코드에서는 monologg Kobert (https://huggingface.co/monologg/kobert) 를 사용합니다.

4. Define Dataset  
   : train.csv 파일을 로드하고, train dataset과 dev dataset을 7:3의 비율로 split합니다. BERTDataset class를 정의하고, tokenizing과 padding을 적용하기 위한 data_collector 를 선언합니다.

5. Define Metric  
   : 대회의 metric인 f1_score를 정의하기 위한 함수를 선언합니다.

6. Train Model Define Model  
   : hyperparameter, metric, data loader를 반영하여 학습할 수 있도록 TrainingArgument 모듈로 training arguments를 정의하고, Trainer 모듈을 사용하여 학습을 진행합니다.

7. Evaluate Model  
   : test.csv 파일을 업로드한 후 data loader에 할당해줍니다. model을 evaluation mode로 변경한 후 model에 넣어 예측값을 뽑아냅니다. 뽑아낸 예측값으로 accuracy를 계산하여 출력합니다.
- requirements.txt : 코드 실행을 위해 필수적으로 설치되어야 하는 패키지의 이름과 그 버전이 적혀있습니다. 본 베이스라인 코드에서는 transformers, sentencepiece, numpy, pandas, evaluate, accelerate, scikit-learn, ipywidgets가 필요합니다.

실행을 위해서는 먼저 pip install -r requirements.txt로 패키지를 먼저 설치한 후, baseline.ipynb의 셀을 순서대로 실행하면 됩니다. 이 떄, BASE_DIR 의 경로를 현재 사용자의 경로로 지정해주시면 됩니다.

코드를 모두 실행하고 나오는 결과물은 BASE_DIR 내의 output.csv 의 파일로 저장됩니다. 이 파일은 기존 test.csv 파일에, 예측 결과가 함께 저장되어 있습니다. 예측 결과의 경우 target 이라는 column에 추가되므로, 기존 test.csv 의 shape이 (9107, 4) 이므로 output.csv 는 (9107,5) 의 shape을 가집니다.

- 추가 팁

- 만약 wandb를 사용하고 싶지 않다면, 학습 전(step 6 이전)에 아래 코드를 실행해주세요!  
  os.environ['WANDB_DISABLED'] = 'true’

Download Data Link

[https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000245/data/data_ (1).tar.gz](https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000245/data/data_%20(1).tar.gz)

Download Baseline Code Link

https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000245/data/code.tar.gz

### 리더보드

![](https://lh6.googleusercontent.com/wcMyA-LNUplbKiuUBqTT6EwE1txLLC44QspGAum-ovwlUilxMjScmm-LAgCqc8799lupPx9PtshxoX4xRXWg6zGI0ZKDNeeF95an2aPD0fgvgWXHCte6DOQqACk6Y5sWTJimZZjKIafzmtGKcuVPw24)

![](https://lh5.googleusercontent.com/FdXK-EUzMgxT8snqt2thrqZkALnH1FDTLY0AXwb-cNyxqVo4CE10cu9K-7Z4BfXzjbx-xv2Hnp1f8kGzavhVnT8I7CAk8EdzoS4LRnTnAX8a4eilTa5uM6A1XmtyptrRw2Axtp4M9Vzf3eYnfWkHeOY)

### 기타 공유 사항

[토론] (공유) Data-Centric AI 기반 다양한 합성 데이터 제작 기법 소개

Posted by 유하늘_조교



양질의 데이터 제작을 위해 human-in-the-loop, human-machine-in-the-loop 등 다양한 시스템이 제안되었는데요. 이와 관련해서는 Human-Centered Artificial Intelligence (HAI)라는 키워드로의 검색을 추천 드립니다. 조금 긴 자료이지만 Stanford의 Institute for HAI에서 출간한 [<On the Opportunities and Risks of Foundation Models>](https://crfm.stanford.edu/report)에서도 관련 내용을 짧게 소개하고 있습니다.

이번 토론 콘텐츠에서는 ChatGPT, LLaMA 등 생성형 AI 기반 대규모 언어 모델 출시 이후 인공지능 모델을 합성 데이터 생성 과정에 도입하는 두 연구를 소개하고자 합니다. 데이터 수집 과정에서 신뢰할 만한 gold label을 구축하는 것은 정말 어렵습니다. 주로 크라우드소싱을 통해 주석 작업자를 모집하는데, 이 과정에서 시간적·금전적으로 큰 비용이 소모됩니다. 대규모 언어 모델로 합성 데이터 샘플 문장을 만드는 것을 넘어, 최근 연구는 대규모 언어 모델을 데이터 라벨링에 도입할 것을 제안하고 있습니다. 과연 ChatGPT 등 대규모 언어 모델이 데이터 주석(라벨링) 작업에서 인간 크라우드소싱 작업자를 대체할 수 있을까요?

# GPT-3

### Want To Reduce Labeling Cost? GPT-3 Can Help (Wang et al., 2021)

[[arXiv]](https://arxiv.org/abs/2108.13487) [[ACL Anthology]](https://aclanthology.org/2021.findings-emnlp.354/)

![](https://lh5.googleusercontent.com/dogVzscuNBnl-NdAFDdrUbXyO9OF5FrXRjzwzKzXVB670S0QnLTuUVc4ELkjPuG7ic2IwdAmAyx7xSa6KLCdi62HflFDmn3yXixslGVGDP2eK4ee8OPEWRG977BQH6LYO0Q23iccYUqOP2H5zyqBSqM)

![](https://lh3.googleusercontent.com/bPWuCgpB1nBdwCET1zQ0M_ce9qSyNgEV6JBmvr5GZITeBBznTTAjOyrCXdF8V4L8BEOuMROX-CayAcBwOevIinQIBO42pPXmhHWk5AFV-9Kf6_JoVnvNguzwrv6w1e_5kQTAtFJo87sMDwEumbzzoSc)

이 논문은 ACL-Findings 2021에 게재된 것으로, 7강에서 잠깐 다뤘으니 이번 토론 콘텐츠에서는 가볍게만 언급하겠습니다. 이 논문에서는 GPT-3을 데이터 주석(라벨링) 작업에 활용할 것을 제안했는데요. 동일 성능 기준으로 인간 크라우드소싱 작업자에만 의존하던 기존 시스템에 비해 약 50%~96%의 비용을 절감했다고 합니다. 하지만 GPT-3은 사람을 능가하는 성능을 보이지는 않았기에, 이후 human-machine-in-the-loop 등의 시스템을 활용하여 데이터를 구축하는 후속 연구가 많이 나오게 되었습니다.

# 

# GPT-3.5

### ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks (Gilardi et al., 2023)

[[arXiv]](https://arxiv.org/abs/2303.15056)

![](https://lh6.googleusercontent.com/-41kfJ6i0VKPMN-UencZyd5ct_CM6K-ZsSTefKwSqcafbWULSi5oiSesBNbi1hz-XplYQ6S1A9r4_YMS8cZHyV2t7Sh--3dkjAVhXbTaHdnI26tdidlI_Gt9r0qRbCavWfm3Mhe-iQy31reNC3EIhdA)

이 논문에서는 약 2천 건의 트윗을 레이블 된 데이터로 만드는 과정에서 크라우드소싱 인간 작업자와 ChatGPT를 비교합니다. 크라우드소싱을 활용한 데이터 수집 목적의 가장 대표적인 플랫폼인 Amazon사의 Mechanical Turk (AMT, MTurk 등으로 불림)에서 인간 작업자를 모집하여 실험을 진행했다고 합니다. 실험 결과, ChatGPT(gpt-3.5-turbo)가 인간 작업자를 정확도와 Inter-Annotator Agreement (IAA) 측면에서 모두 능가했다고 하네요. ChatGPT를 사용한다면 시간과 비용을 모두 줄일 수 있다는 점을 고려한다면 매우 희망적인 결과입니다. 데이터 라벨링에서 ChatGPT의 사용이 크라우드소싱 인간 작업자 고용에 비해 약 20배 가량 저렴하다고 하네요. 다만, 이 실험에서 다룬 태스크는 모두 단순한 분류 문제였다는 점에서 다른 태스크에서의 활용 가능 여부 점검이 필요하다고 생각합니다.

ChatGPT의 출시 이후 ChatGPT(gpt-3.5-turbo)의 데이터 라벨링 성능에 대해 연구한 논문이 많이 나오고 있습니다. 소개해드린 논문과 비슷한 주제를 다룬 [“AnnoLLM: Making Large Language Models to Be Better Crowdsourced Annotators (He et al., 2023)”](https://arxiv.org/abs/2303.16854)도 참고하세요.

### Is ChatGPT better than Human Annotators? Potential and Limitations of ChatGPT in Explaining Implicit Hate Speech (Huang et al., 2023)

[[ACM DL]](https://dl.acm.org/doi/abs/10.1145/3543873.3587368) [[arXiv]](https://arxiv.org/abs/2302.07736)

![](https://lh4.googleusercontent.com/KnKXvbWoYifCfRp11b0qlkF7KwyCG0ye_oWrBsq4if1IaQUpO5IzyZATxvC-_tycVoqx6aQUjGd6cQwXg6aHzc0jDR4OuoVszbZY54enwIFq0Pnjh9-a4vYUaVCZUEOR9AEDjBdJpj6rbOLAZByikGo)

이 논문은 ACM WWW 2023 Companion에 게재된 것으로, Implicit Hate Speech Explanation의 레이블 생성에서 ChatGPT를 활용할 수 있는지에 대해 다룹니다. 트윗을 기반으로 구축한 내재적 혐오 표현 데이터셋인 [LatentHatred 데이터셋](https://aclanthology.org/2021.emnlp-main.29/)을 바탕으로 (1) ChatGPT가 내재적 혐오 표현을 감지할 수 있는지와 (2) 내재적 혐오 표현에 대해 정보성과 명료성 측면에서 양질의 설명을 생성할 수 있는지를 평가했습니다. 실험 결과, ChatGPT는 약 80%의 샘플에서 정확한 합성 데이터를 생성했다고 합니다.

# 

# GPT-4

### Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark (Pan et al., 2023)

[[arXiv]](https://arxiv.org/abs/2304.03279)

![](https://lh6.googleusercontent.com/zN7Don8zg2YAn35ePZ8V-yF6sxa2M9fDkg9snCmVjjCHspqszxRGhZ8Quo3YS3e3lU1m_GVagrhFbsHfseXZOOhw6z05qG3p5lecWFU7eIP7B392jarDznCyfxbhr-rX3-dAMmD3JoQl7djX2GZieWM)

이 논문은 ICML 2023 Oral에 게재된 것으로, 2023년 3월에 출시된 GPT-4를 데이터 주석 작업에 활용했습니다. 이 논문에서는 MACHIAVELL라는 시스템을 제안하여 GPT-3.5나 GPT-4와 같은 대규모 언어 모델을 활용하여 “먼저 설명하고 나서 라벨링하는 (explain-then-annotate” 방식을 제안합니다. 강화학습, 윤리적 프롬프트 등 여러 기법을 도입하여 데이터 주석 작업의 정확도를 높이고자 했는데요. 이 논문에 따르면 GPT-4로 생성한 레이블은 시간당 $25를 받는 숙련된 인간 크라우드소싱 주석 작업자보다 정확했다고 합니다.

# 논의점

상기 논문을 읽고 제가 생각한 논의점은 다음과 같습니다. 이에 대한 캠퍼 분들의 의견을 댓글로 자유롭게 나눠주세요. 새로운 논의점을 제안해주셔도 좋습니다.

1. 제가 소개해드린 연구는 모두 영어 데이터셋 구축에서 ChatGPT를 활용하는 것에 대한 내용을 다루고 있습니다. 과연 ChatGPT(또는 다른 고성능 대규모 언어 모델)를 영어 외의 다른 언어의 데이터셋 구축에 도입할 수 있을까요? 저자원 언어에 대해서는 어떨까요? (저자원 언어가 무엇인지 잘 모르신다면, [The State and Fate of Linguistic Diversity and Inclusion in the NLP World (Joshi et al., 2020)](https://aclanthology.org/2020.acl-main.560/)을 참고하세요!)

2. 자연어 처리 하위 태스크의 종류에 따라 데이터 주석 작업의 난이도 역시 다릅니다. 또한, hate speech detection, bias detection, morality judgement 등 사람의 주관적인 판단을 요구하는 태스크일 수록 Inter-Annotator Agreement (IAA)가 낮아지죠.  
   “Is ChatGPT better than Human Annotators? Potential and Limitations of ChatGPT in Explaining Implicit Hate Speech (Huang et al., 2023)”에 따르면, ChatGPT가 Hate Speech Explanation Generation에서 약 80%의 일치도를 보였다고 하는데요. 과연 다른 주관적인 태스크에서도 ChatGPT를 활용하는 것이 가능할까요? 특히, 언어·문화적 이해를 필요로 하는 태스크 및 예시는 어떨까요?

3. 더 정확한 합성 데이터 생성을 위해 조정할 수 있는 ChatGPT의 변수나 매개변수는 무엇이 있을까요? 일례로, 첫 번째 논문에서는 temperature 0.2와 1의 결과를 함께 비교하였습니다.

4. 캠퍼 분들이 시도해보신 프롬프트 중 합성 데이터 생성에 도움이 되는 프롬프트가 있었나요? ChatGPT를 이용해 데이터 레이블을 생성할 경우, 길고 장황한 답변이 아닌 내가 원하는 형식의 답변을 이끌어내는 여러분만의 팁이 있었나요? Zero-shot으로 하되 instruction이나 태스크에 대한 설명을 철저히 할 때보다 Few-shot으로 예시를 줄 때 더 좋은 성능을 보였나요? Few-shot 프롬프트에서 예시 샘플에 대한 편향을 보이지는 않았나요?  

마지막으로, 이번 토론 콘텐츠에서 소개해드린 연구 이외에도 정말 다양한 Data-Centric AI 기반 합성 데이터 제작 기법이 고안되고 있는데요. 캠퍼 분들께서 알고 계시는 다양한 연구를 댓글이나 게시글로 공유해보면 좋을 것 같습니다. 이상입니다.



[토론] [공유] Data-Centric AI기반 다양한 데이터 증강 기법 소개

Posted by 심미단_조교

2023.05.10.09:48



딥러닝 모델을 학습시키는 목적이나 task에 따라, 학습 데이터가 충분하지 않은 경우가 존재합니다. 이런 경우를 ‘low-resource setting’ 이라 표현합니다. 충분한 양,질의 데이터는 성능 향상을 위한 중요한 역할을 하기 때문에, 학계에서는 이러한 low resource setting를 극복하기 위한 다양한 연구가 이루어지고 있습니다. 그 중에서도 데이터 자체의 양을 늘림으로서 충분한 분량의 학습 데이터를 확보하는 데이터 증강(data augmentation)은 low resource setting에서 많이 사용되는 기법 중 하나입니다.

컴퓨터 비전 분야에서는 기존 이미지 데이터를 회전시키는 image rotation, 이미지 중 일부를 잘라내는 cropping 등의 방법으로 데이터 증강을 수행합니다. 그렇다면 자연어에서의 데이터 증강은 어떤 방법을 사용할까요?

본 토론 콘텐츠에서는 데이터 증강 기법 세가지(Back-Translation, Round Trip Translation, Iterative Back-Translation)를 소개해드리고자 합니다. 세가지 방법 모두 기계 학습(Machine Translation) 분야에서 제안되었습니다.

### 1. Back-Translation (BT)

Back-translation이란 target language sentence를 바탕으로 source language sentence를 생성하는 과정이며, 아래 논문에서 처음으로 제안되었습니다.

[Improving Neural Machine Translation Models with Monolingual Data](https://aclanthology.org/P16-1009.pdf), ACL 2016

지금부터는 구체적인 수식 보다 예시를 들어 자세히 설명해보겠습니다. 한국어 벤치마크인 KLUE-TC(Topic Classification)의 학습 데이터 중 “유튜브 내달 2일까지 크리에이터 지원 공간 운영” 이라는 텍스트 데이터를 영어로 번역하면 아래와 같습니다.

![](https://lh3.googleusercontent.com/xUNI3phQPw5XeebqErAwQpJBNoD8C548ftik8mQv59b8iwDh1Pbp42o3Tn6KlpfTMvGNopseheCoyzOLEzA6lyvbXv7ztxLTkcrPGJ5C9o0WIZSpNr4dL0CiCx8Joz-ln_LIMpxl9lm4LgOGQ2VNx-g)

원래의 source lanaguage 데이터로부터 “YouTube operates a creator support space until the 2nd of next month”라는 target language인 영어 데이터를 얻을 수 있습니다. 이제 이 글을 다시 한국어로 번역해보면 어떨까요?

![](https://lh6.googleusercontent.com/8e4GQJSwZWIqFobQrS57wBdREzGbKY4wd9oDzRe0hd9LW892EqHgI2OzxucF8DVIzeHYNqtuXFpqKblKY7TN9KFoYHNNV1ggjLgZ3AyD7slvm142p3VtuQv6q5fAfcEjjWyjxVV2PQKQburxAAAEbSY)

이와 같이, “유튜브 내달 2일까지 크리에이터 지원 공간 운영”과 의미는 같지만 다른 단어로 표현된 “유튜브는 다음 달 2일까지 크리에이터 지원 공간을 운영한다.” 라는 문장이 생성됩니다.

google 번역 뿐 아니라 api를 제공하는 다른 번역기를 사용할 수 있습니다. 또한 한국어-영어-한국어 뿐 아니라 한국어-일본어-한국어 등 다양한 언어를 사용하여 데이터 증강이 가능합니다.

이 방법을 사용하여 실제로 데이터를 생성해보면, 원래의 source data보다는 비교적 정제되지 않다는 느낌을 받을 수 있습니다. 본 논문에서는 이렇게 noise가 많을수록 학습 성능의 향상을 확인할 수 있었습니다.

### 2. Iterative Back-Translation

Iterative Back-Translation 이란 이름 그대로 ‘반복적으로 수행하는(Iterative) Back-Translation’ 입니다.

[Iterative Back-Translation for Neural Machine Translation](https://aclanthology.org/W18-2703/), ACL 2018

위 논문은 Iterative Back-Translation 을 제안한 논문이며, 아래 그림은 논문에서 가져온 Figure입니다. Re-Back-Translation 즉 여러번의 Back-Translation을 그림으로 확인할 수 있습니다. 

![](https://lh6.googleusercontent.com/bYnOQ9oFYOFHG3jhuWgx-4flT6kgKnL5nFLiq48iqmQ4-rKeMkB-qOpODhG8D-RZHPYyT8jyeNRYzvLCHMNYv34iGIRQuqy6G5Cqr2w7Au6TsXAl4XJZmZ-p0HfpT2LNSIeZ-uG4DXwB0lGCh3wJ4Bo)

논문에서는 기존 Back-Translation의 데이터 퀄리티를 지적하고, Back-Translation을 여러번 반복함으로써 low-resource setting 뿐 아니라 high-resource setting에서도 좋은 성능을 달성했음을 보였습니다.

특히, 아래의 두 Table들은 Back-Translation과 Iterative Back-Translation의 성능을 비교함으로써 Iterative Back-Translation의 뛰어남을 확인할 수 있습니다. 

![](https://lh4.googleusercontent.com/-ivBxWikiz4DyJhF8HbLRpNb_cjsHAJ-DI1YZvhNLzRw_3mo1LS8Dz22odDwAJnFXpNDM6XOkO0wVE4CbgVf2rPh4j9aGF0opjzG2sD6QAkxa9MYDmNC0Pi7Z2GQHWyHtaEBwJI3PDpBP4hDIflWImE)

![](https://lh5.googleusercontent.com/CvhkQRftqJ2ha12MGkVRm4GuTHk9MwWOWcIKjNA0v70XqKKXHiHTQt_uKtnn070CsZfkk88eJY4wu2eGLa-EiW2xLw8vKiQXzaOLKxZdlTQHN7ra2HYnwoMGNQT8TYhiQdYwuKlwHPOrsPjG8fUJrHQ)

### 3. Round Trip Translation (RTT)

Back-Translation 또는 Iterative Back-Translation이 단일 언어(monolingual)에서의 low-resource issue를 해결하기 위해서였다면, 마지막으로 소개해드릴 Round Trip Translation은 bilingual 데이터가 부족한 학습 환경을 완화하기 위해 아래 논문에서 제안되었습니다.

[Augmenting Neural Machine Translation through Round-Trip Training Approach](https://www.degruyter.com/document/doi/10.1515/comp-2019-0019/html), Open Data Science 2019

 Round Trip Translation은 Back-and-forward translation이라고도 알려져 있으며, 두 단계의 translation이 close-loop을 구성합니다. 다시 말해, back translation system과 forward translation system이 서로의 피드백을 반복적으로 반영하여 학습됩니다.

두 개의 translation system A, B이 학습되는 절차는 아래와 같습니다.

1. initial small bilingual data로 translation system A, B를 각각 학습시킴

2. translation system A가 source language data → target language data 의 번역을 수행함 (forward-translation)

3. translation system B가 target language data → source language data 의 번역을 수행함 (backward-translation)

4. (2)와 (3)에서의 translation accuracy를 계산한 후, 이를 바탕으로 reward score를 계산함

5. Stochastic Gradient Descent으로 (4)를 maximize하도록, 반복적으로 A,B를 update함

논문에서 공개한 아래 테이블은, round-tripping을 적용하기 전후의 성능 차이입니다. 모든 평가 지표에서 round-tripping의 적용이 확연한 성능 향상을 이끌었음을 확인할 수 있습니다. 

![](https://lh6.googleusercontent.com/4vu-A_GIWjZ8aBLORkzHJkhhtzOsqRy50b44Ej7BHg1-NAkixw--n5rW4TmvWTOTyIM_6aWccwBAP1t7NH_DvzMHW95kB25iF8j8r_s9D04FFo2lpA3yehS7-dhct7xP0IVHvP2sZ3asX7x15AA5PLk)

---

소개해드린 세가지 기법 중, 가장 간단한 Back-Translation을 KLUE-TC 학습 데이터에 적용한 코드는 아래와 같습니다.

from datasets import load_dataset # pip install datasets  
import googletrans # pip install googletrans==4.0.0-rc1  
import pandas as pd  

def BTS(input_str, translator):

    # target language로의 번역 수행 

    temp_result = translator.translate(inputstr, dest='en').text 

    # 다시 source language로의 번역 수행 

    result = translator.translate(temp_result, dest='ko').text

    return result

# 사용할 번역기 api 로드

translator = googletrans.Translator()

# 증강할 데이터 로드

dataset = load_dataset('klue', 'ynat')  

# data frame으로 변경

ids, titles, labels, urls, dates = [],[],[],[],[]

for data in dataset['train']: 

    ids.append(data['guid'])

    titles.append(data['title'])

    urls.append(data['url'])

    labels.append(data['label'])

    dates.append(data['date'])  

df = pd.DataFrame({'id':ids, 'input_text':titles, 'label': labels, 'url': urls, 'date':dates})  

augmented_data, augmented_ind, wrong_ind = [], [], []

for i in range(len(df)):

    try: 

        data = df['input_text'].iloc[i]

        augmented = BTS(data, translator)

        augmented_data.append(augmented)

        augmented_ind.append(i)

    except: 

        wrong_ind.append(i)

df = df.iloc[augmented_ind]

df['input_text'] = augmenteddata

df.to_csv('augmenteddataset.csv', index = False)

이렇게 증강한 데이터를 학습에 적용할 수 있습니다 !!

---

소개해드린 세가지 데이터 증강 기법에 대해 아래와 같은 논의점을 제안드립니다. 이에 대한 캠퍼 분들의 의견을 댓글로 자유롭게 나눠주세요. 새로운 논의점을 제안해주셔도 좋습니다.

### 논의점

- Back-translation으로 생성한 데이터를 별도의 전처리 없이 바로 사용해도 괜찮을까요? 예를 들어, back-translation의 결과가 original data와 같은 경우의 존재 가능성을 배제할 수 없습니다.

- 증강한 데이터의 양과 원래의 데이터의 양의 비율에 따라 성능이 달라질 수 있을까요? 두 데이터의 양이 (1) 동일한 경우, (2) 원래의 데이터가 더 많은 경우, (3) 증강한 데이터가 더 많은 경우 중에서 어떤 경우가 제일 성능이 좋을까요? 그렇게 생각하신 이유 및 근거도 궁금합니다.

그 외에도 데이터 증강에서의 고민이나 다른 데이터 증강 방법론 등 공유하고 싶은 내용이 있으시면 토론 게시판에 마구마구 작성해주세요:)



[토론] (공유) Data-Centric AI 기반 Label Error Detection 기법 소개

Posted by 박가연_조교

2023.05.10.10:45

 

흔히 사용하는 데이터셋인 IMDB(영화 리뷰 데이터셋)나 Amazon Reviews 데이터셋을 이용하여 모델을 구현할 때, 클래스 라벨은 모두 옳게 라벨링이 되어있을 것이라 가정하고 성능 향상을 위해 모델을 수정하는 경우가 많습니다. 하지만 아래의 경우처럼 라벨링이 잘못 되어 있는 경우도 상당 부분 존재합니다. Amazon Reviews 데이터셋의 예시인데요, 주어진 문장인 ‘기분이 나아지고 더 이상 우울하지 않다’는 긍정적인 내용에 ‘negative’로 잘못 라벨링이 되어있는 것을 확인할 수 있습니다. 잘못 라벨링된 데이터로 학습할수록, 아무리 모델을 발전시켜도 일정 성능 이상의 향상을 이루기 어렵습니다.

![](https://lh6.googleusercontent.com/7BXjuhgiA_I6W0XASSvujj2SWLrINFafTGRoD7_BinIjrh-JdhuAGOEvJQjgk95DbWRkeiHAku0oScE6t7l0tnGuM7HjtwmycE9FcIp_p3zN6CojAa_JgkpJKeo5robJXOB-bbe48GQdsmX-HL07JZI)

출처: [https://labelerrors.com/](https://labelerrors.com/)

이번 토론 시간에는 data-centric AI 패키지인 Cleanlab을 이용해 라벨링 이슈를 탐지해볼  예정입니다. Cleanlab은 내가 학습시킨 모델을 이용하여 수정할 수 있는 데이터셋 이슈를 발견한 다음, 이를 이용하여 더 나은 모델을 학습할 수 있게 합니다.

Cleanlab은 Confident Learning(CL) 알고리즘을 통해 잘못된 데이터셋의 라벨을 식별합니다. CL은 라벨 노이즈의 불확실성을 통해 데이터셋에 노이즈가 있어도 잘 학습될 수 있도록 하는 supervised learning과 weak-supervision의 하위 분야라고 할 수 있습니다. CL의 전반적인 프로세스는 다음과 같습니다.

1. 노이즈 데이터 정제하기(pruning noisy data)

2. 노이즈 추정 계산하기(counting to estimate noise)

3. 학습을 위한 예제 순위 매기기(ranking examples to train with confidence)  

아래 그림은 CL의 프로세스 및 예시인데요, 먼저 내가 학습시킨 모델을 통해 주어진 노이즈 라벨(y ̃)과 잠재적인 정답 라벨(y*) 간 결합 분포를 직접 추정합니다. 이때 모델은 성능을 낼 수 있는 어떤 모델이라도 사용이 가능합니다. 이렇게 데이터셋의 특징을 통해 라벨링 이슈를 파악한 후, 노이즈 라벨을 제거해주게 됩니다. 

![](https://lh4.googleusercontent.com/MJLgAFmN0PlRxn-rzHMo2xkCAdkbZXXkLG1fY8KO2Gyt1sB3EyaqHY6HcCFhwdMl9HcViR6-4lBDFr8_1KzMtiB-w-W1qvtUZ03ccD4HEwZlMTX02tA6iYLOLPgIS8_7wFBBD22EWPRe7CfANigrw0c)

그럼 이제 이번 대회에서 사용하는 KLUE 데이터와 KoBERT 모델을 이용하여 데이터셋 이슈를 파악해 보겠습니다. 먼저 데이터셋 내에 어떤 라벨 에러가 존재하는지 알아보도록 하겠습니다.

아래 코드는 모든 모델에 적용가능한 코드로, label 데이터(labels)와 정답 예측 확률(predprobs) 정보를 통해 잘못 라벨링되어 있을 가능성이 높은 순으로 정렬된 데이터를 얻을 수 있습니다. 이때 사용한 라벨은 학습 데이터셋에 대한 라벨이며, predprobs는 KoBERT의 정답 예측 확률입니다. 

출력된 결과를 통해 정치(0) , 경제(1) , 사회(2) , 생활문화(3) , 세계(4) , IT과학(5) , 스포츠(6) 의 7개의 주제 중 주어진 문장이 과연 잘못 라벨링이 되어 있는지 살펴보도록 하겠습니다. 먼저 첫번째 문장은 정치보다 IT과학에 가깝다고 할 수 있습니다. 두번째 문장과 세번째 문장도 정치보다는 각각 사회와 생활문화 에 가깝다고 할 수 있습니다. 

---

from cleanlab.filter import find_label_issues  

ordered_label_issues = find_label_issues(  
    labels=dataset_train['target'], # 데이터셋 라벨  
    pred_probs=train_pred_probs, # 정답 예측 확률  
    return_indices_ranked_by='selfconfidence',  
)  

head_issues=ordered_label_issues[:3]  
for issue in head_issues:  
    print('input text:',dataset_train.iloc[issue]['input_text'])  
    print('label:',dataset_train.iloc[issue]['label_text'])

    print(‘------------------’)

![](https://lh4.googleusercontent.com/81zaJlOLV-NlTxxRJnl566P9KJPJLt1xVSEFkk7uBttI9lJ0ICgqjPaXkV7hsJfv3BAuXDqWSG5c0x8a1wfhJbmM1UtBiyTDRK--fmS6nkU65U48SaCaLMyMvHCw3QaMQmb6iU57s9bAV2dlF5MRpmc)

---

다음은 데이터셋 라벨별로 이슈를 확인할 수 있는 코드입니다. 앞의 코드와 마찬가지로 label 데이터와 정답 예측 확률이 필요하고 추가적으로 라벨 정보도 필요합니다. 코드 실행 결과 클래스 별 라벨 노이즈와 라벨의 품질 점수를 확인할 수 있습니다. 앞에서 확인한 3개의 예시 모두 정치(0) 분야에서 나왔는데요, 분석 결과 정치(0)의 라벨 노이즈가 가장 심하고, 라벨의 퀄리티도 가장 낮은 것을 확인할 수 있습니다.

---

from cleanlab.dataset import health_summary  
class_names=[0,1,2,3,4,5,6]  
health_summary(dataset_train['target'], train_pred_probs, class_names=class_names)

![](https://lh4.googleusercontent.com/3IAxYDEJZvAK7dNUks-TXOWs_j3iggxE8pBltCTaRnrZsQe8Q-TlNvgrur86jzl3RavAU5Vsm7jK5KqDlUR1qMfiTee0NOYjVrb6TkeZU1TAoTbmx7MKuXlkDYHzNb6Ru0EdWf41lB755fMAqcu5mvs)

---

이렇게 데이터셋 내 에러를 찾는 것 외에도 cleanlab은 아웃라이어를 탐지하거나, 겹치는 클래스를 알려주는 등 다양한 기능을 제공하고 있습니다. [cleanlab documentation](https://docs.cleanlab.ai/stable/index.html)을 참고하셔서 다양한 기능을 체험해본 후 직접 적용해보시면 좋을 것 같습니다. 

이번 토론에서 논의해볼 사항은 다음과 같습니다. label error를 통해 데이터를 정제하는 과정에서 과연 노이즈를 어느 정도까지 제거해야 성능이 잘 나올까요? 잘못된 라벨링이라고  판단된 모든 데이터를 지우면 성능이 올라갈까요? 노이즈와 데이터의 균형 관점에서 캠퍼분들이 생각하시는 최적의 기법이 무엇인지, 노이즈를 어디까지 허용하면 좋을지 등 다양한 의견을 공유해주세요~!



[토론] (공유) KLUE 데이터셋 소개 및 KLUE를 인용한 다양한 한국어 자연어처리 연구 소개

Posted by 김채형

2023.05.10.17:57

 

## KLUE 데이터셋 소개

![](https://lh3.googleusercontent.com/dYwiyIJrww9X2ukqzxsbkqe3jAXd1S2Ukq_8XJbIT7MHwbGgOMmQIM6EkJC7sJ-zolLbXXAp1KnbDimO409y5HjZOMTxPMDmpcYuuZIpXw5mw9WRhEi5BSQ6yriUykhbFERgmyTxf9G3rW9nbUuo47o)

딥러닝 기반의 인공지능 모델을 만들기 위해 가장 중요한 것 중 하나는 바로 데이터입니다. 하지만 한국어의 경우 저자원 언어에 속해 활용 가능한 데이터가 영어에 비해 현저히 적습니다. 이에 따라 KLUE(Korean Language Understanding Evaluation)가 제안되었는데요. KLEU는 한국어 자연어처리 분야의 대표적인 데이터셋 중 하나로, 다양한 자연어처리 task, 예를 들어 주제 분류, 개체명 인식, 자연어 추론 등을 포함하고 있습니다. 우리는 KLUE 벤치마크를 통해 다양한 모델을 학습하고 평가할 수 있게 되었습니다. 따라서 이번 토론 콘텐츠에서는 KLUE 데이터셋을 소개하고, KLUE를 인용한 한국어 자연어처리 연구 두 가지를 소개하려고 합니다.

![](https://lh6.googleusercontent.com/y2XXuOeJDDxMkSfhkz_Pnz6jaNTSyyegEHqoeI2wJLJ5ga4RJRsL-21LaxMmvFifNXgLEsO5FMlj2Io9Dw7T1ScsKg5kL5HLwOxwDy8dDINDGy7cK4drqxoejp78T3CSBSTcrDDBZjcPMoeZdQRH3yY)

### 1. 주제 분류 (Topic Classification; TC)

![](https://lh6.googleusercontent.com/tnTDYQd3n6I4hG0pBvSFx33jVwkG4cvMibXdp48063xKGWa2PywbhKnp8TcJ3DdaIrEg4ahy0VwGvRfLAYaj7dz1GuwiJEu2MMggM7RqXia0myq1bgdgd3mY-ChsuHdUYenmfSpGgsQO_7P0Fy4wyPs)

주제 분류(Topic Classification; TC)의 목표는 주어진 텍스트의 주제를 분류하는 것입니다.

KLUE-TC에서 사용된 데이터는 2016년 1월부터 2020년 12월까지 네이버 뉴스에 올라간 연합뉴스의 헤드라인입니다. 이 헤드라인을 가지고 해당 텍스트가 정치, 경제, 사회, 문화, 세계, IT/과학, 스포츠 총 7가지 주제 중 어떤 카테고리에 속하는지를 예측합니다.

해당 task를 평가할 때에는 macro F1 score이 사용됩니다. multi-class classification에 대한 F1 score를 구할 때에는 각 label/class에 대한 F1 score를 구한 뒤 이를 평균을 내는 방식으로 계산합니다.

### 2. 의미론적 유사도 (Semantic Textual Similarity; STS)

의미론적 유사도(Semantic Textual Similarity; STS)의 목표는 주어진 두 개의 텍스트 간의 의미적인 유사도를 측정하여 수치로 표현하는 것입니다.

KLUE-STS에서 사용된 데이터는 에어비앤비 리뷰, 정책 뉴스 브리핑 자료, 스마트 홈 기기와의 대화입니다.

에어비앤비 리뷰와 정책 뉴스 브리핑 자료 같은 경우에는 문장 간의 유사성을 추정하기 힘든 경우가 있었는데, 이때는 네이버 파파고를 사용하여 영어로 번역했다가 다시 한국어로 번역하여 유사한 문장 쌍을 생성했다고 합니다. 이러한 기법을 Round-Trip Translation(RTT)라고 부르는데, 이렇게 함으로써 원래 문장의 핵심 의미를 유지하면서 어휘 표현이 살짝 다른 문장을 생성할 수 있었다고 합니다.

해당 task를 평가할 때에는 F1 score와 피어슨 상관 계수를 사용합니다. F1 score의 경우, 유사도 3.0을 기준으로 유사하다/유사하지 않다로 레이블링 한 후 F1 score를 구합니다.

### 3. 자연어 추론 (Natural Language Inference; NLI)

자연어 추론(Natural Language Inference; NLI)의 목표는 전제(premise)로 주어진 텍스트와 가설(hypothesis)로 주어진 텍스트 간의 관계를 추론하는 것입니다.

KLUE-NLI에서 사용된 데이터는 위키트리, 정책 뉴스 브리핑 자료, 위키뉴스, 위키피디아, 네이버 영화 리뷰, 에어비앤비 리뷰입니다. 전제와 가설 간의 관계는 가설이 참인 경우(entailment), 가설이 거짓인 경우(contradiction), 가설이 참일 수도 있고 아닐 수도 있는, 즉 알 수 없는 경우(neutral)로 레이블링 되어있습니다.

해당 task를 평가할 때에는 정확도(Accuracy)를 사용합니다.

### 4. 개체명 인식 (Named Entity Recognition; NER)

![](https://lh3.googleusercontent.com/GdqTF6wSlrRicw1lbCjAWa-sbMS4ppH57HNUX15DsjoHgO4Cmoas82CoyR4oaMlBzAoZHGOKwh5rRCw935EWGARbXpWvyPXE6Vbr7E20hq3UWLo8JXkJdM9TeqyBGeyjEObgvb-blRBBTGyB-T4yhp4)

개체명 인식(Named Entity Recognition; NER)의 목표는 주어진 텍스트에서 개체의 경계를 감지하고 개체의 유형을 분류하는 것입니다. 다시 말해 어떤 이름을 가진 단어를 보고 그 단어가 어떤 유형인지를 분류하는 것입니다. 예를 들어 “OpenAI는 2015년에 설립되었다”라는 문장이 주어지면 모델은 OpenAI(조직), 2015년(시간)을 출력해야 합니다.

KLUE-NER에서 사용된 데이터는 위키트리(formal한 텍스트)와 네이버 영화 리뷰(informal한 텍스트)입니다. 이 데이터들은 사람(PS), 위치(LC), 기관(OG), 날짜(DT), 시간(TI), 수량(QT) 총 6가지의 개체명으로 레이블링 되어있습니다.

해당 task를 평가할 때에는 entity-level macro F1 score와 character-level macro F1 score를 사용합니다.

### 5. 관계 추출 (Relation Extraction; RE)

![](https://lh5.googleusercontent.com/KBVhNi8_XeC0j98Ts5WkP9ye7g900rTj9Sn8DZ0OYxyeD01DKdQRn6VbY8YwYSLjzge-jB9rvvuK04qkKmXg-ZnmUK3c0qFVA0c7H_7GyZQ6ikN0_GWyHUd4QFaNWPCLwacd7U_iMAU0gFJ7rvvF_So)

관계 추출(Relation Extraction; RE)의 목표는 주어진 텍스트에 나타난 개체(entity)즉 단어들 간의 의미론적 관계를 추론하는 것입니다. 예를 들어 “스티브 잡스는 샌프란시스코에서 태어났다”라는 문장에서 “스티브 잡스”와 “샌프란시스코”의 관계는 “placeofbirth”라고 볼 수 있습니다.

KLUE-RE에서 사용된 데이터는 위키피디아, 위키트리, 정책 뉴스 브리핑 자료입니다. 이 데이터들은 “norelation”, “per:dateof_birth” 등 총 30개의 관계로 레이블링 되어있습니다.

해당 task를 평가할 때에는 micro F1 score와 AUC(Area Under the Precision-Recall Curve)를 사용합니다.

### 6. 의존 구문 분석 (Dependency Parsing; DP)

의존 구문 분석(Dependency Parsing; DP)의 목표는 단어 간의 관계 정보를 찾는 것입니다. 다시 말해 단어와 단어 간의 관계를 기본으로 누가 head인지, 의미적으로 지배하는지/지배당하는지에 대한 관계 정보를 찾는 것입니다.

의존 구문 분석기는 dependency 문법에 기반하여 주어진 문장의 그래프 구조를 예측합니다. 이러한 구문 분석 트리는 일반적으로 dependent를 head에 연결하는 dependency arc와 dependent와 head 사이의 관계를 나타내는 dependency label로 구성됩니다. 예를 들어, “철수가 사과를 먹었다”라는 문장이 주어질 때, “철수가”는 “먹었다”의 dependent이며 주어의 관계를 가집니다. 이러한 관계를 DEPREL(dependency relation classes)이라고 하는데 DEPREL은 NP(Noun Phrase), VP(Verb Phrase)와 같은 9개의 Syntax 태그와 SBJ(Subject), OBJ(Object)과 같은 6개의 Function 태그의 조합으로 이루어진 36개의 TTA Dependency를 따릅니다.

KLUE-DP에서 사용된 데이터는 위키트리와 에어비앤비 리뷰입니다.

해당 Task를 평가할 때에는 UAS(Unlabeled Attachment Score)와 LAS(Labeled Attachment Score)가 사용됩니다. UAS는 HEAD만을 평가 대상으로 삼고, LAS는 HEAD와 DEPREL을 모두 평가 대상으로 삼습니다.

### 7. 기계 독해 (Machine Reading Comprehension; MRC)

![](https://lh5.googleusercontent.com/lWzrgysmYpCnb5lYXtxOlU8MHlbmImz3glOJEaAFg5GSQlx6KwA6dKLPu5z9nyL7pL_jCUfR25faFcq2uCql3ZMmai8WB6IYZ74DAKwuUZy4sOvfxqDf565nD6oGpVcUampPtCHxZ5YS5USEbUp0vuw)

기계 독해(Machine Reading Comprehension; MRC)의 목표는 본문와 질문이 주어질 때 답을 찾는 것입니다.

KLUE-MRC에서 사용된 데이터는 위키피디아, 아크로팬, 한국경제신문입니다.

해당 task를 평가할 때에는 EM(Exact Match)과 ROUGE(character-level ROUGE-W)를 사용합니다.

### 8. 대화 상태 추적 (Dialogue State Tracking; DST)

![](https://lh6.googleusercontent.com/rVXmOxFF1bHagYhWrxM1lSlV2RicdVze4zX6xDZKo7YGxBHCz_bM5uL9CgRQIloZoKroFfxf9Iyojttcj_SWdLY2PykJhjSM87ssATg5nhDOzsEQ_4yQXTnjI5APIL4Ei13AuRiQWlnF80-rtadUgHo)

대화 상태 추적(Dialogue State Tracking; DST)의 목표는 사용자와 시스템 간의 목적 지향형 대화에서 대화 상태를 예측하는 것입니다. 구체적으로, 목적 지향형 대화 시스템에서 각 턴마다 사용자의 요청 혹은 사용자가 준 정보를 key-value로 바꾸는 것을 대화 상태(dialogue state)라고 합니다. 예를 들어, “서울 중앙에 있는 박물관을 찾아주세요”라는 사용자의 발화가 주어지면 종류-박물관, 지역-서울 중앙을 예측하는 것입니다.

KLEU-DST에서 사용된 데이터는 호텔, 레스토랑, 관광정보, 택시, 지하철 총 5가지 domain을 다룹니다.

해당 task를 평가할 때에는 JGA(Joint Goal Accuracy)와 slot F1 score가 사용됩니다.

## KLUE를 인용한 다양한 한국어 자연어처리 연구 소개

KLUE 데이터셋과 KLUE를 기반으로 한 다양한 연구들이 한국어 자연어처리 분야에서 활발하게 진행되고 있습니다. 이 중 Data-Centric NLP와 밀접한 관련이 있는 “Rethinking Annotation: Can Language Learners Contribute?” 논문과 “Schema Encoding for Transferable Dialogue State Tracking” 논문을 소개하고자 합니다.

### Rethinking Annotation: Can Language Learners Contribute? (ACL 2023) [[link](https://arxiv.org/abs/2210.06828)]

전통적으로 연구자들은 벤치마크 데이터셋에 대한 annotation을 진행하고자 할 때 native speaker를 모집했습니다. 그러나 native speaker를 모집하는 것이 어려운 언어가 있으며, 이러한 언어를 배우는 사람들이 데이터에 annotation을 하도록 하면 도움이 될 것입니다. 본 논문에서는 언어 학습자가 벤치마크 데이터셋에 annotation을 제공할 수 있는지 여부를 조사했습니다. 먼저 36명의 언어 학습자를 모집하고, 두 가지 유형의 추가 자원(사전 및 기계 번역 문장)을 제공하고, 그들의 언어 숙련도를 측정하기 위해 미니 테스트를 수행했습니다. 이때 3개의 언어(영어, 한국어, 인도네시아어)와 4개의 NLP task(감성 분석, 자연어 추론, 개체명 인식, 기계 독해)를 대상으로 하였습니다. 본 논문에서는 언어 학습자, 특히 중급 또는 고급 언어 능력을 가진 사람들이 추가 자원의 도움을 받아 상당히 정확한 레이블을 제공할 수 있다는 것을 발견했습니다. 또한, 데이터 annotation이 어휘와 문법 측면에서 학습자의 언어 능력을 향상시킨다는 것을 증명했습니다. 이러한 연구 결과는 언어 학습자를 포함하도록 annotation 작업을 확장하면 native speaker를 모집하기 어려운 언어에 대한 벤치마크 데이터셋을 구축할 수 있는 기회를 열 수 있다는 의미를 가집니다.

### Schema Encoding for Transferable Dialogue State Tracking (COLING 2022) [[link](https://arxiv.org/abs/2210.02351)]

대화 상태 추적(Dialogue State Tracking; DST)은 task-oriented 대화 시스템의 필수적인 요소입니다. 최근 연구는 DST를 위한 deep neural model에 초점을 맞추고 있습니다. 하지만 neural model은 훈련을 위해 대규모의 데이터셋을 필요로 합니다. 또한 neural model은 일반적으로 주어진 데이터셋을 모방하도록 훈련되기 때문에 다른 domain에 적용하려면 새로운 데이터셋이 필요합니다. 본 논문에서는 새로운 domain으로의 효과적인 transfer를 위한 neural DST 방법론인 Schema Encoding for Transferable Dialogue State Tracking (SET-DST)을 제안합니다. transferable DST는 target domain에 대한 데이터셋이 거의 없어도 대화 시스템 개발을 지원할 수 있습니다. 스키마 인코더를 사용하여 데이터셋을 모방할 뿐만 아니라 데이터셋의 스키마를 이해합니다. 본 논문에서는 새로운 스키마를 인코딩하고 multi-domain setting의 DST에 사용하여 모델을 새 domain으로 trasnfer하는 것을 목표로 합니다. 그 결과, SET-DST는 MultiWOZ 2.1에서 joint accuracy를 1.46 point 향상시켰습니다.

## 논의점

- KLEU-STS 데이터를 확보하기 위해 저자들은 Round-Trip Translation(RTT) 기법을 사용하였는데요. Data-Centric NLP 강의에서 배운 다양한 augmentation 방법 중 KLEU-STS에 적용할 수 있는 또다른 방법이 있을까요?

- KLEU-STS 데이터를 확보하기 위해 사용된 Round-Trip Translation(RTT) 기법을 KLEU 내 다른 task에 대해서도 활용할 수 있을까요?

- Rethinking Annotation 논문에서는 non-native annotator들이 감성 분석, 자연어 추론, 개체명 인식, 기계 독해 총 4개의 task에 대해 충분한 annotation 능력을 가짐을 보였습니다. 일반적으로 각각의 자연어처리 task들은 서로 다른 난이도를 가집니다. 같은 summarization task라고 하더라도 abstractive summarization이 extractive summarization보다 더 어렵다고 여겨지는 것처럼 말이에요. 그렇다면 KLEU에 존재하는 다른 task들에 대해서는 어떨까요? 위 4개의 task를 제외한 다른 task들에 대해서도 non-native speaker들이 충분히 뛰어난 annotation 능력을 보일 수 있을까요?







## 프로젝트 팀에서진행된 규칙

증강, 필터링 등의 작업을 통해 변경된 데이터셋은 각자의 개인폴더에 versioning 후 업로드해야 한다.

1. 변경된 데이터셋 버전에 대한 정보는 개인 폴더의 `README.md` 파일 생성 후 간략하게라도 작성해야 한다.

## 프로젝트 팀 Directory

```
README.md
data
├── sample_submission.csv
├── test.csv 
└── train.csv
hgkim
├── your_files
├── your_folders
└── ...
hwyoo  
├── your_files
├── your_folders
└── ...
krseong    
├── your_files
├── your_folders
└── ...
sphong    
├── your_files
├── your_folders
└── ...
thkim
├── your_files
├── your_folders
└── ...
README.md
```
