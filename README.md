# Rationale Extraction for Sentiment Classification

## Background

### Interpretability 
As deep learning revolutionizes the field of NLP, researchers are also calling for interpretability. 
Interpretability is often used to confirm other important desiderata which are hard to codify, such as fairness, trust, and causality ([Doshi-Velze and Kim, 2017](https://www.semanticscholar.org/paper/Towards-A-Rigorous-Science-of-Interpretable-Machine-Doshi-Velez-Kim/5c39e37022661f81f79e481240ed9b175dec6513)).
The term "interpretability" should be used with caution though as interpretability is not a monolithic concept, but in fact reflects several distinct ideas ([Lipton, 2018](https://www.semanticscholar.org/paper/The-Mythos-of-Model-Interpretability-Lipton/d516daff247f7157fccde6649ace91d969cd1973)).
Two large aspects of interpretability often discussed in NLP are *faithfulness* and *interpretability* ([Wiegreffe and Pinter, 2019](https://www.semanticscholar.org/paper/Attention-is-not-not-Explanation-Wiegreffe-Pinter/ce177672b00ddf46e4906157a7e997ca9338b8b9); [Jacovi and Golberg, 2020](https://www.semanticscholar.org/paper/Towards-Faithfully-Interpretable-NLP-Systems%3A-How-Jacovi-Goldberg/579476d19566efc842929ea6bdd18ab760c8cfa2); [DeYoung et al., 2020](https://www.semanticscholar.org/paper/ERASER%3A-A-Benchmark-to-Evaluate-Rationalized-NLP-DeYoung-Jain/642038c7a49caa9f0ac5b37b01fab5b2b8d981d5)).
*Faithfulness* refers to how accurately it reflects the true reasoning process of the model, while *plausibility* refers to how convincing the interpretation is to humans ([Jacovi and Golberg, 2020](https://www.semanticscholar.org/paper/Towards-Faithfully-Interpretable-NLP-Systems%3A-How-Jacovi-Goldberg/579476d19566efc842929ea6bdd18ab760c8cfa2)).

### Rationale
Interpretation can have many different forms, but the most common in NLP is *rationale*. Rationale is a text snippet from the input text. It is analoguous to saliency map in computer vision.

### Motviation of Rationale Extraction
Some works use attention ([Bahdanau et al., 2015](https://www.semanticscholar.org/paper/Neural-Machine-Translation-by-Jointly-Learning-to-Bahdanau-Cho/fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5)) to obtain rationales for interpretation. 
However, it is argued that attentions cannot serve as a faithful interpretation ([Jain and Wallace, 2019](https://www.semanticscholar.org/paper/Attention-is-not-Explanation-Jain-Wallace/1e83c20def5c84efa6d4a0d80aa3159f55cb9c3f); [Brunner et al., 2020](https://www.semanticscholar.org/paper/On-Identifiability-in-Transformers-Brunner-Liu/9d7fbdb2e9817a6396992a1c92f75206689852d9)).  
Some works use post-hoc methods ([Ribeiro et al., 2015](https://www.semanticscholar.org/paper/%22Why-Should-I-Trust-You%22%3A-Explaining-the-of-Any-Ribeiro-Singh/5091316bb1c6db6c6a813f4391911a5c311fdfe0); [Sundararajan et al., 2017](https://www.semanticscholar.org/paper/Axiomatic-Attribution-for-Deep-Networks-Sundararajan-Taly/f302e136c41db5de1d624412f68c9174cf7ae8be)), but these methods also do not ensure faithfulness ([Lipton, 2018](https://www.semanticscholar.org/paper/The-Mythos-of-Model-Interpretability-Lipton/d516daff247f7157fccde6649ace91d969cd1973); [Adebayo et al., 2022](https://openreview.net/forum?id=xNOVfCCvDpM)).

Meanwhile, some claims that strictly faithful interpretation is impossible.  

> *Explanations must be wrong. They ***cannot have perfect fidelity*** with respect to the original model.*   
[Rudin, 2019](https://www.semanticscholar.org/paper/Stop-explaining-black-box-machine-learning-models-Rudin/bc00ff34ec7772080c7039b17f7069a2f7df0889)  

> *...we believe ***strictly faithful interpretation is a 'unicorn'*** which will likely never be found.*   
[Jacovi and Golberg, 2020](https://www.semanticscholar.org/paper/Towards-Faithfully-Interpretable-NLP-Systems%3A-How-Jacovi-Goldberg/579476d19566efc842929ea6bdd18ab760c8cfa2)

This is where *rationale extraction* comes in. Instead of attributing the model's decision to a certain part of the input, this framework extracts the important part of the input and use it as the *only* input to the prediction model. In this way, no matter how complex the prediction model is, we can gurantee that the extracted input is fully faithful to the model's prediction.

### Rational Extraction

Extracting the rationale, which is essentially assigning a binary mask to every input token, is non-differentiable. To tackle this problem, reinforcement learning was initially s
([Lei et al., 2016](https://www.semanticscholar.org/paper/Rationalizing-Neural-Predictions-Lei-Barzilay/467d5d8fc766e73bfd3e9415f75479823f92c2f7)),
while more recent works employed reparameterization ([Bastings et al., 2019](https://www.semanticscholar.org/paper/Interpretable-Neural-Predictions-with-Binary-Bastings-Aziz/8c5465eb110d0cab951ca6858a0d51ae759d2f9c); [Paranjape et al., 2020](https://www.semanticscholar.org/paper/An-Information-Bottleneck-Approach-for-Controlling-Paranjape-Joshi/c9aeb7e31b16b7273a80ae748b3ff48105928147#references)). Another work has bypassed this problem by decoupling the extraction and prediction, achieving a competitive performance with an ease of use ([Jain et al., 2020](https://www.semanticscholar.org/paper/Learning-to-Faithfully-Rationalize-by-Construction-Jain-Wiegreffe/922e6e3bafe38a712597c05d3a907bd10763b427?sort=is-influential)).

## Overview

Succeeding my last project, [KR3: Korean Restaurant Review with Ratings](https://github.com/Wittgensteinian/kr3), I applied rationlae extraction on sentiment classification. Please note that these dataset and task are probably not the most ideal setting for rationale extraction (See Additional Note for more details).  
   
I followed the framework **FRESH(Faithful Rationale Extraction from Saliency tHresholding)**, proposed by [Jain et al., 2020](https://www.semanticscholar.org/paper/Learning-to-Faithfully-Rationalize-by-Construction-Jain-Wiegreffe/922e6e3bafe38a712597c05d3a907bd10763b427?sort=is-influential).   

![FRESH](./images/FRESH.png)  
FRESH could be divided into three parts.  
1. Train *support model* end-to-end on the original text.
1. Use *feature scoring method* to obtain soft(continuous) score, and then discretize it with selected *strategy*.  
1. Train *classifier* on the label and the extracted rationale.

Since FRESH is a framework, many design choices could be made. These are some of the choices in the paper, with choice in this project marked **bold**. 
- Support Model: **BERT**
- Feature Scoring Method: **attention**, gradient 
- Strategy: **contiguous**, **Top-*k***
- Classifier: **BERT**

## Result


## Additional Note


## Examples


## References

