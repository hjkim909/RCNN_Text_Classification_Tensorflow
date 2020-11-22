# RCNN_Text_Classification_Tensorflow
 [RCNN](http://zhengyima.com/my/pdfs/Textrcnn.pdf) implementation using tensorflow

## Requirements
- tensorflow
- pandas
- nltk
- sklearn

## Dataset
AG News Dataset [Download](https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms)



## Training
```
python main.py --epoch  10
```

## Result

|Accuracy|F1-score|
|------|---|
|0.9172|0.9181|

## Reference
- Lai, S., Xu, L., Liu, K., & Zhao, J. (2015, February). Recurrent convolutional neural networks for text classification. In Twenty-ninth AAAI conference on artificial intelligence.[Link](http://zhengyima.com/my/pdfs/Textrcnn.pdf)
- jungwhank's Pytorch implementation. [Link](https://github.com/jungwhank/rcnn-text-classification-pytorch)
