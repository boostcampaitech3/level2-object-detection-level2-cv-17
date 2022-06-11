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
