# MMDetection

### configs file using directions (Korean)
- configs 디렉토리 안의 model, dataset, schedule 각각의 폴더 안에 사용하고자 하는 config 설정을 한다.
- _base_ 디렉토리 안의 파일들에는 model, dataset, schedule, runtime에서 설정한 config를 가져온다.해당 .py 파일들은 train, test 과정에서 load한다.
- 모든 config 세팅은 앞서 말한 것 처럼 _base_파일에서 병합된다. 이 때, 추가적으로 config를 설정하고 싶거나 수정하고 싶다면 _base_에서 overwrite를 해주면 된다.

### configs file using directions (English)
- Construct your model, dataset, schedule at each folder
- _base_ is total config file which load at train, test process. 
- All config constructed is merge in _base_ and if you want additionally modify config, overwrite config in file which is in _base_.

## Train
### train usage
```
usage: train_refactor.py [-h] [--config CONFIG] [--workdir WORKDIR]
                         [--no-validate] [--seed SEED]
                         [--tags TAGS [TAGS ...]] [--kfold] [--no-kfold]
                         [--fold FOLD] [--options OPTIONS [OPTIONS ...]]
                         [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]

Train a detector

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       train config file path
  --workdir WORKDIR     the root dir to save logs and models about each
                        experiment
  --no_validate         whether not to evaluate the checkpoint during training
  --seed SEED           random seed
  --tags TAGS [TAGS ...]
                        record your experiment speical keywords into tags list
                        --tags batch_size=16 swin_cascasdedont use white space
                        in specific tag
  --kfold               wheter use K-fold Cross-Validation
  --no-kfold
  --fold FOLD           if no kfold cross validation, you must set fold number
  --options OPTIONS [OPTIONS ...]
                        override some settings in the used config, the key-
                        value pair in xxx=yyy format will be merged into
                        config file (deprecate), change to --cfg-options
                        instead.
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-
                        value pair in xxx=yyy format will be merged into
                        config file. If the value to be overwritten is a list,
                        it should be like key="[a,b]" or key=a,b It also
                        allows nested list/tuple values, e.g.
                        key="[(a,b),(c,d)]" Note that the quotation marks are
                        necessary and that no white space is allowed.
```

### the key options (Korean)

- --config : command에서 반드시 옵션으로 사용하시는 total train config file의 경로를 넣어줘야 합니다.
- --no-validate : 해당 optoin을 주시면, validation을 수행하지 않습니다. 따라서, all dataset으로 학습시키고 싶으시다면, 해당 옵션을 주시면 전체 데이터셋으로 학습하도록 로직이 짜여져 있습니다. ( default는 즉 옵션을 주지 않는다면 validatoin 수행)
- --tags : wandb에 로깅할 때, 추가적인 특이사항들을 적어두기 위해 tag설정을 해줍니다. 키워드들을 넣어주시면 됩니다. 예시는 help에 적혀 있습니다. ex) python train.py --tags batch_size=16 swin_cascade
- --kfold & --no-kfold : kfold를 하시고 싶으시다면 --kfold를 옵션으로 주시면 됩니다.(defualt설정) 만약, k-fold를 하시고 싶으시지 않으시다면, --no-kfold를 옵션으로 주시면 됩니다. 이 때, k-fold를 하지 않는다면 어떤 폴드를 사용할 것인지를 정해줘야하기에 --fold에 몇번 폴드를 사용할 것인지를 추가적으로 옵션을 주셔야합니다.

### the key options (English)

- --config : you must option your train config file path
- --no-validate : if you option in command, your code script won't validation. So, If you want train with all dataset you should option
- --tags : more easily check your model special note write your keywords down
- --kfold & --no-kfold : if you want to kfold, option the formmer but you don't want kfold , you should option the latter and specify which fold with --fold option.

### options & cfg-options usage
해당 사용 내용은 help에 자세하게 적혀 있습니다. 간단히 적자면 cfg dict의 값들에 접근해서 config의 옵션을 command line에서 바꿔주기 위한 설정들입니다. 

### Train comand 
- basic  
`python train.py --config ./configs/_base_/cascade_rcnn_swin224_pafpn_1x.py `
- more example  
`python train.py --config ./configs/_base_/cascade_rcnn_swin224_pafpn_1x.py --cfg-options checkpoint_config.max_keep_ckpts=3 runner.max_epochs=30`

## Test
### test usage
```
usage: test_refactor.py [-h] [--config CONFIG]
                        [--checkpoint_path CHECKPOINT_PATH]
                        [--work_dir WORK_DIR] [--TTA] [--out OUT]
                        [--fuse-conv-bn] [--eval EVAL [EVAL ...]] [--show]
                        [--show-dir SHOW_DIR]
                        [--show-score-thr SHOW_SCORE_THR]
                        [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
                        [--options OPTIONS [OPTIONS ...]]
                        [--eval-options EVAL_OPTIONS [EVAL_OPTIONS ...]]

MMDet test (and eval) a model

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       test config file path
  --checkpoint_path CHECKPOINT_PATH
                        ex) exp11/Fold1/epoch_20.pth
  --work_dir WORK_DIR   the directory to save the file containing evaluation
                        metrics
  --TTA                 TTA work
  --out OUT             output result file in pickle format
  --fuse-conv-bn        Whether to fuse conv and bn, this will slightly
                        increasethe inference speed
  --eval EVAL [EVAL ...]
                        evaluation metrics, which depends on the dataset,
                        e.g., "bbox", "segm", "proposal" for COCO, and "mAP",
                        "recall" for PASCAL VOC
  --show                show results
  --show-dir SHOW_DIR   directory where painted images will be saved
  --show-score-thr SHOW_SCORE_THR
                        score threshold (default: 0.3)
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-
                        value pair in xxx=yyy format will be merged into
                        config file. If the value to be overwritten is a list,
                        it should be like key="[a,b]" or key=a,b It also
                        allows nested list/tuple values, e.g.
                        key="[(a,b),(c,d)]" Note that the quotation marks are
                        necessary and that no white space is allowed.
  --options OPTIONS [OPTIONS ...]
                        custom options for evaluation, the key-value pair in
                        xxx=yyy format will be kwargs for dataset.evaluate()
                        function (deprecate), change to --eval-options
                        instead.
  --eval-options EVAL_OPTIONS [EVAL_OPTIONS ...]
                        custom options for evaluation, the key-value pair in
                        xxx=yyy format will be kwargs for dataset.evaluate()
                        function
```
### the key options (Korean)

- --config : command에서 반드시 옵션으로 사용하시는 total train config file의 경로를 넣어줘야 합니다.
- --checkpoint_path : 학습시키면서 생기는 detector의 weigth의 경로
- --TTA : TTA를 수행하고 싶다면 해당 옵션을 주시면 됩니다.

### the key options (English)

- --config : you must option your train config file path
- --checkpoint_path : detector weights path ( generated checkpoint path during training )
- --TTA :  if you option in command, your test do TTA

### options & cfg-options usage
해당 사용 내용은 help에 자세하게 적혀 있습니다. 간단히 적자면 cfg dict의 값들에 접근해서 config의 옵션을 command line에서 바꿔주기 위한 설정들입니다.

### Test command
- basic  
`python test.py --config ./configs/_base_/cascade_rcnn_swin224_pafpn_1x.py`
- more example  
`python test.py --config ./configs/_base_/cascade_rcnn_swin224_pafpn_1x.py --checkpoint_path /opt/ml/detection/level2-object-detection-level2-cv-17/mmdet/work_dirs/exp31/Fold3/epoch_30.pth --cfg-options data.test.piepline.1.img_scale="(224,224)" --TTA`


