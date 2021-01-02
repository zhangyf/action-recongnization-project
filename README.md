# Action recognization project

## This is the official github repo for project. All timelines will be posted here. Check if you need.

### Timeline

#### 2020/12/23 repo initiated.
#### 2020/12/23 online real time inference code uploaded. Need to conduct testing.
#### 2020/12/24 Pass test for online inference
#### 2020/12/25 Switched to new framework mmaction
#### 2020/12/25 Waiting for complete training data
#### 2020/12/28 Setup initial video preprocessor for mmaction video processing
#### 2020/12/31 Initial evaluation completed with TSN model. Evaluation accuracy : 90%
#### 2021/01/01 Initial offline inference completed. Check at provided location for inference video.
#### 2021/01/01 Complete training/testing code uploaded to github. Currently support recognization for all available data until now. 
#### 2021/01/02 Documentation available

### Priority table

#### 1.实时性的问题，是否可以修改下已经部署在服务器的那版的参数，我们来验证下1s内是否OK。如果OK的话，这个问题就可以close了。 [Marked as solved partially on Dec 24th]
#### 2. json还请帮忙搞一下，并提供个时间点 [Marked as solved partially on Dec 24th]
#### 3. 文档在这三个问题里相对最不紧急，可以先搞定前面2个，最后再给这个

### Plan for the project

#### 1. Add multi-thread for online inference code. This should support the pipeline for current inference stage. Pay attention as min frame enforced as 25. Time the code if possible.
#### 2. Modify json output based on discussion on 12/31. Now given the softmax probability and video path together with prediction.
#### 3. completion of 1,2 satisfies priority table item 1,2.

