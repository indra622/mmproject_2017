# mmproject_2017


Caffe와 CNN을 활용한 자동차 차종 분류기 개발

## caffemodeling

caffe에서 사용하는 모델의 prototxt파일과 solver, labeling 정보를 담은 directory

deploy.prototxt
 - 모델 테스트용 CNN구조를 표현한 prototxt 파일

label.txt
 - 각 클래스의 이름을 나타내는 파일

mmtest.sh, mmtrain.sh
 - 각각 학습과 테스트를 위해 만들어진 bash 기반 shell 파일

solver100_46_11.prototxt
 - caffe solver 파일

train_val100_46_11.prototxt
 - 학습용 neural network구조를 표현한 prototxt 파일


## web_client

Web demo용으로 사용하는 python flask 기반 demo client

app.py
 - web demo의 실행 코드 및 파일

exifutil.py
 - input image를 관리하는 모듈

spectro.py
 - wav파일을 spectrogram으로 변환해 주는 모듈
