## Recycle Trash Object Detection Competition
&nbsp;&nbsp;쓰레기 대란, 매립지 부족 등의 사회문제 해결에 기여하고자 대회 형식으로 진행되었던 프로젝트이며, 이미지에서 10종류의 쓰레기(일반 쓰레기, 종이, 금속, 유리, 플라스틱 등)를 검출하는 모델을 설계하였습니다. 이는 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다.      
- Input: 이미지, 쓰레기 종류, 쓰레기 bbox 좌표 (coco format)    
- Output: 쓰레기 종류, 쓰레기 bbox 좌표, Confidence (score)       

![image](https://user-images.githubusercontent.com/39791467/173186400-4571dc1d-05b2-4195-b172-b38ee62b56bf.png)

## 💁TEAM
### CV 17조 MG세대
|민선아|백경륜|이도연|이효석|임동우|
| :--------: | :--------: | :--------: | :--------: | :--------: |
|<img src="https://user-images.githubusercontent.com/78402615/172766340-8439701d-e9ac-4c33-a587-9b8895c7ed07.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766371-7d6c6fa3-a7cd-4c21-92f2-12d7726cc6fc.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172784450-628b30a3-567f-489a-b3da-26a7837167af.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766321-3b3a4dd4-7428-4c9f-9f91-f69a14c9f8cc.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766404-7de4a05a-d193-496f-9b6b-5e5bdd916193.png" width="120" height="120"/>|
|[@seonahmin](https://github.com/seonahmin)|[@baekkr95](https://github.com/baekkr95)|[@omocomo](https://github.com/omocomo)|[@hyoseok1223](https://github.com/hyoseok1223)|[@Dongwoo-Im](https://github.com/Dongwoo-Im)|
|Detectron2 기반의</br>모델 설계|1 Stage Model|2 Stage Model|2 Stage Model,</br>Augmentation|2 Stage Model</br>with Pseudo Labeling|

## 🏆Leaderboard Ranking
- Public: 9등, mAP 0.7008
- Private: 9등, mAP 0.6873

## 🧪Experiment
1. Reproducibility: seed 고정
2. Augmentation: Heavy augmentation, Mosaic augmentation
3. 학습 전략: K-Fold Cross Validation, All Dataset
4. Multi-scale 전략: Multi-scale augmentation, TTA MultiscaleFlip
5. Pseudo labeling

## ✨Model
1-stage, 2-stage model을 이용해 모델의 다양성을 높이고 앙상블 효과를 얻고자 했습니다. 각 모델에 대한 실험과 점수를 시각화 했습니다.
### 1-stage detector - Yolov5
<img src = "https://user-images.githubusercontent.com/39791467/173186840-f3e1d15f-bd40-45ff-a3ba-43c0ad5d8b4d.png" width="800"></img>

### 2-stage detector - Cascade RCNN
![image](https://user-images.githubusercontent.com/39791467/173186868-6cf74ce2-f108-40eb-a6ce-43c7b52ac670.png)

## 🌈Ensemble
### 앙상블 전략
다양한 앙상블 전략을 통해 앙상블을 진행했습니다.      
1. 각각의 모델을 한 번에 WBF로 앙상블한다.       
2. 1번과 달리 여러 개의 모델 중 일부를 SoftNMS나 WBF로 Ensemble해서 생성된 중간 Ensemble 결과를 최종 Ensemble에 사용한다.      
3. 분할 정복 알고리즘과 유사하게 모델들을 그룹지어 각각 SoftNMS Ensemble을 통해 중간 결과를 생성하고, 그 결과를 최종적으로 모아 WBF로 앙상블한다.      

<img src = "https://user-images.githubusercontent.com/39791467/173186813-dadb85c7-1fc2-4091-8eab-bd3edb33de3e.png" width="800"></img>

## 📖 Reference
* mmdetection
    * github : https://github.com/open-mmlab/mmdetection
    * documentation : https://mmdetection.readthedocs.io

* yolov5
    * github : https://github.com/ultralytics/yolov5
