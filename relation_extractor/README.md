# Relation Extractor

관계 추출을 실제로 수행하는 모듈들을 포함하는 패키지. 많은 코드들이 대회에서 제공된 Baseline과 huggingface의 transformers를 기반으로 하고 있음.

## trainer.py

Pre-trained 모델을 불러오고 모델을 정의하고 학습을 수행하는 과정까지 포함된 모듈

## data_loader

제공된 데이터를 불러와서 pre-processing 및 토큰화를 수행하는 모듈

## predictor.py

학습이 완료된 모델을 불러와서 추론을 수행하고 submission.csv 결과 파일을 생성하는 모듈