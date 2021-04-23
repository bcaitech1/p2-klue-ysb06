# Dataset Hub

데이터셋 생성을 위한 데이터 수정을 위한 코드

## analyzer.py

원래는 데이터 분석만을 위한 코드였지만 https://github.com/machinereading/kor-re-gold의 데이터를 사용할 수 있게 레이블 변경하고 데이터 오류를 수정하는 코드로 변경됨.

## simple_upsampler.py

anlyzer 모듈을 대체하여 단순하게 부족한 레이블에 대한 작은 양의 upsampling을 수행하는 코드