# models
## classification.py
二分类模型：基于特征表示对数据应用多层感知器进行二分类


loss：


loss=keras.losses.categorical_crossentropy #[1,0],[0,1]


loss='binary_crossentropy' #1,0

## Chinese_preprocessing.py
中文分词工具：jieba和thulac

## audio.py
音频处理：清洗脏数据，提取音频特征表示，音频切割
