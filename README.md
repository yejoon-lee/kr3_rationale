# Rationale Extraction in Sentiment Classification

## Abstract
Many real-world scenarios demand the interpretability of NLP models. Despite various interpretation methods, it's often hard to guarantee those interpretations to truly reflect the model's decision. Meanwhile, rationale extraction first **extracts** the relevant part of the input (rationale), and then feed it as the **only** input to the predictor. This ensures the **strict faithfulness** of the rationale.    
  
In this project, I implemented FRESH ([Jain et al., 2020](https://www.semanticscholar.org/paper/Learning-to-Faithfully-Rationalize-by-Construction-Jain-Wiegreffe/922e6e3bafe38a712597c05d3a907bd10763b427?sort=is-influential)) for rationale extraction in sentiment classification, using [KR3: Korean Restaurant Review with Ratings](https://github.com/Wittgensteinian/kr3). Discontiguous rationale attained a comparable performance to the original paper, easily outperforming the random baseline. Unfortunately, this wasn't true for contiguous rationale. Plus, it would be worth noting that rationale extraction could excel better in more appropriate settings. This discussion is elaborated in the section *Limits*.  
  
[Presentation slides](https://drive.google.com/file/d/19iwwN2DYzpCxZh95uYb4O07TV3fuBH1h/view?usp=sharing) used in the class.

## Background

### 🔍 Interpretability 
As deep learning revolutionizes the field of NLP, researchers are also calling for interpretability. 
Interpretability is often used to confirm other important desiderata which are hard to codify, such as fairness, trust, and causality ([Doshi-Velze and Kim, 2017](https://www.semanticscholar.org/paper/Towards-A-Rigorous-Science-of-Interpretable-Machine-Doshi-Velez-Kim/5c39e37022661f81f79e481240ed9b175dec6513)).
The term "interpretability" should be used with caution though as interpretability is not a monolithic concept, but in fact reflects several distinct ideas ([Lipton, 2018](https://www.semanticscholar.org/paper/The-Mythos-of-Model-Interpretability-Lipton/d516daff247f7157fccde6649ace91d969cd1973)).
Two large aspects of interpretability often discussed in NLP are *faithfulness* and *plausibility* ([Wiegreffe and Pinter, 2019](https://www.semanticscholar.org/paper/Attention-is-not-not-Explanation-Wiegreffe-Pinter/ce177672b00ddf46e4906157a7e997ca9338b8b9); [Jacovi and Golberg, 2020](https://www.semanticscholar.org/paper/Towards-Faithfully-Interpretable-NLP-Systems%3A-How-Jacovi-Goldberg/579476d19566efc842929ea6bdd18ab760c8cfa2); [DeYoung et al., 2020](https://www.semanticscholar.org/paper/ERASER%3A-A-Benchmark-to-Evaluate-Rationalized-NLP-DeYoung-Jain/642038c7a49caa9f0ac5b37b01fab5b2b8d981d5)).
*Faithfulness* refers to how accurately the interpretation reflects the true reasoning process of the model, while *plausibility* refers to how convincing the interpretation is to humans ([Jacovi and Golberg, 2020](https://www.semanticscholar.org/paper/Towards-Faithfully-Interpretable-NLP-Systems%3A-How-Jacovi-Goldberg/579476d19566efc842929ea6bdd18ab760c8cfa2)).

### 🗯️ Rationale 
Interpretation can have many different forms, but the most common in NLP is *rationale*. Rationale is a text snippet from the input text. It is analoguous to saliency map in computer vision.

### 🔥 Motviation of Rationale Extraction 
Some works use attention ([Bahdanau et al., 2015](https://www.semanticscholar.org/paper/Neural-Machine-Translation-by-Jointly-Learning-to-Bahdanau-Cho/fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5)) to obtain rationales for interpretation. 
However, it is argued that attention cannot provide a faithful interpretation ([Jain and Wallace, 2019](https://www.semanticscholar.org/paper/Attention-is-not-Explanation-Jain-Wallace/1e83c20def5c84efa6d4a0d80aa3159f55cb9c3f); [Brunner et al., 2020](https://www.semanticscholar.org/paper/On-Identifiability-in-Transformers-Brunner-Liu/9d7fbdb2e9817a6396992a1c92f75206689852d9)).  
Some works use post-hoc methods ([Ribeiro et al., 2015](https://www.semanticscholar.org/paper/%22Why-Should-I-Trust-You%22%3A-Explaining-the-of-Any-Ribeiro-Singh/5091316bb1c6db6c6a813f4391911a5c311fdfe0); [Sundararajan et al., 2017](https://www.semanticscholar.org/paper/Axiomatic-Attribution-for-Deep-Networks-Sundararajan-Taly/f302e136c41db5de1d624412f68c9174cf7ae8be)), but these methods also do not ensure faithfulness ([Lipton, 2018](https://www.semanticscholar.org/paper/The-Mythos-of-Model-Interpretability-Lipton/d516daff247f7157fccde6649ace91d969cd1973); [Adebayo et al., 2022](https://openreview.net/forum?id=xNOVfCCvDpM)).

Meanwhile, some claims that strictly faithful interpretation is impossible.

> *Explanations must be wrong. They ***cannot have perfect fidelity*** with respect to the original model.*   
[Rudin, 2019](https://www.semanticscholar.org/paper/Stop-explaining-black-box-machine-learning-models-Rudin/bc00ff34ec7772080c7039b17f7069a2f7df0889)  

> *...we believe ***strictly faithful interpretation is a 'unicorn'*** which will likely never be found.*   
[Jacovi and Golberg, 2020](https://www.semanticscholar.org/paper/Towards-Faithfully-Interpretable-NLP-Systems%3A-How-Jacovi-Goldberg/579476d19566efc842929ea6bdd18ab760c8cfa2)

This is where *rationale extraction* comes in. Instead of attributing the model's decision to a certain part of the input, this framework first **extracts** the important part of the input and then feed it as the **only** input to the prediction model. In this way, no matter how complex the prediction model is, we can guarantee that the extracted input is **fully faithful** to the model's prediction. Hence, this framework is known to be *faithful-by-construction* ([Jain et al., 2020](https://www.semanticscholar.org/paper/Learning-to-Faithfully-Rationalize-by-Construction-Jain-Wiegreffe/922e6e3bafe38a712597c05d3a907bd10763b427?sort=is-influential)).

### 🔦 Rationale Extraction 

Extracting the rationale, which is essentially assigning a binary mask to every input token, is non-differentiable. To tackle this problem, using reinforcement learning was initially proposed
([Lei et al., 2016](https://www.semanticscholar.org/paper/Rationalizing-Neural-Predictions-Lei-Barzilay/467d5d8fc766e73bfd3e9415f75479823f92c2f7)),
while more recent works employed reparameterization ([Bastings et al., 2019](https://www.semanticscholar.org/paper/Interpretable-Neural-Predictions-with-Binary-Bastings-Aziz/8c5465eb110d0cab951ca6858a0d51ae759d2f9c); [Paranjape et al., 2020](https://www.semanticscholar.org/paper/An-Information-Bottleneck-Approach-for-Controlling-Paranjape-Joshi/c9aeb7e31b16b7273a80ae748b3ff48105928147#references)). Another work has bypassed this problem by decoupling the extraction and prediction, achieving a competitive performance with an ease of use ([Jain et al., 2020](https://www.semanticscholar.org/paper/Learning-to-Faithfully-Rationalize-by-Construction-Jain-Wiegreffe/922e6e3bafe38a712597c05d3a907bd10763b427?sort=is-influential)).


## Project

Succeeding my last project, [KR3: Korean Restaurant Review with Ratings](https://github.com/Wittgensteinian/kr3), I applied rationlae extraction on sentiment classification.   

> ⚠️ Please note that these dataset and task are probably not the most ideal setting for rationale extraction (See the section *Limits* for more details).  
   
I followed the framework **FRESH(Faithful Rationale Extraction from Saliency tHresholding)**, proposed by [Jain et al., 2020](https://www.semanticscholar.org/paper/Learning-to-Faithfully-Rationalize-by-Construction-Jain-Wiegreffe/922e6e3bafe38a712597c05d3a907bd10763b427?sort=is-influential).   

![FRESH](./images/FRESH.png)  
FRESH could be divided into three parts.  
1. Train *support model* end-to-end on the original text.
1. Use *feature scoring method* to obtain soft(continuous) score, and then discretize it with selected *strategy*.  
1. Train *classifier* on the label and the extracted rationale.

Since FRESH is a framework, many design choices could be made. Following are some of the choices in the paper, with choices in this project marked **bold**. 
- Support Model: **BERT**
- Feature Scoring Method: **attention**, gradient 
- Strategy: **contiguous**, **Top-*k***
- Classifier: **BERT**



## Result

Among the two aspects of interpretability, faithfulness doesn't have to be evaluated. This is due to the nature of *faithful-by-construction* framework. The other aspect, plausibility, requires human evaluation which I couldn’t afford. Results below only show the performance of the model, which is also crucial. If the prediction is faithful but wrong, what’s the point?

### Notes
- All the performance metrcs are **macro F1**, so *higher-the-better*.
- All the metrics regarding to *Jain et al., 2020* are reported in the paper, i.e. not reproduced.

### **Top-*k*** rationales extracted from attention were **better than random** rationales.
However, for contiguous rationales, there was no significant difference.

| Method                | F1 |
|-----------------------|:-----------:|
| Full text             |     .93     |
| Top-*k*; attention      |   **.82**   |
| Contiguous; attention |     **.75**     |
| Top-*k*; random         |     .74     |
| Contiguous; random    |     .74     |

###  **Top-*k*** rationales showed **similar** degree of **performance drop** as in *Jain et al., 2020*
- Compared to *SST*, the performance drop is similar. However, for contiguous rationales, the performance drop was larger. 
- It's worth noting that for *Movies* the drop was marginal, though the rationale ratio is higher. 
- Note that *SST* and *Movies* are chosen among several evaluation datasets from *Jain et al., 2020*, as these are sentiment classification, just like *KR3*.

| Dataset (rationale ratio) |     *Jain* - SST  (20%)    |    *Jain* - Movies (30%)    | *Ours* -     KR3  (20%)    |
|:-------------------:|:----------------:|:------------------:|:---------------:|
| Full text           |           .90    |            .95     |          .93    |
| Top-*k*; attention    |          .81     |            .93     |          **.82**    |
| Contiguous; attention |          .81         |           .94          |          **.75**         |

### **Remaining** text after rationales are extracted are **still powerful**.
- This isn’t desirable, because it shows that rationales are not necessary for the prediction. 
- I suspect this is largely due to the nature of the task, i.e. information is scattered all over the text.

| Method                     | Performance |
|----------------------------|:-----------:|
| Full text                  |     .93     |
| Remaining after Top-*k*      |     .74     |
| Remaining after Contiguous |     .74     |


## Code & Training logs
Order of python scripts:
1. `fresh_support_model.py`
1. `fresh_extract.py`
1. `fresh_tokenize.py`
1. `fresh_train.py`

Training logs: [Weights & Biases](https://wandb.ai/wittgensteinian/kr4?workspace=user-wittgensteinian)


## Limits (truth be told...)

### Unfixed bug🪲 in rationale extraction code  
- Originally 14345 batches, but only 11967 batches had successfully gone through extraction via *contiguous* strategy (cf. 14171 batches for *Top-k*).
- Reduced dataset size in *contiguous* might’ve contributed to the poor performance.
- Why don't you fix the bug? -> in fact... (see the next section)

### KR3 is probably not **the most ideal** setting for rationale extraction
Settings I'd prefer for rationale extraction are
- Long text (*vs* short text)
- Text-based task (*vs* requires common sense/world knowledge) 
- Information needed for the task compressed in a narrow region (*vs* scattered all over)

Examples of more ideal settings would be
- BoolQ ([Clark et al., 2019](https://www.semanticscholar.org/paper/BoolQ%3A-Exploring-the-Surprising-Difficulty-of-Clark-Lee/9770fff7379a7ab9006b48939462354dda9a2053))
- FEVER ([Thorne et al., 2018](https://www.semanticscholar.org/paper/FEVER%3A-a-Large-scale-Dataset-for-Fact-Extraction-Thorne-Vlachos/b1d24e8e08435b7c52335485a0d635abf9bc604c))
- Evidence Inference ([Lehman et al., 2019](https://www.semanticscholar.org/paper/Inferring-Which-Medical-Treatments-Work-from-of-Lehman-DeYoung/6d529cf091a84200d721cdaf5419baf43ee21566))
- ECtHR ([Chalkidis et al., 2021](https://www.semanticscholar.org/paper/Paragraph-level-Rationale-Extraction-through-A-case-Chalkidis-Fergadiotis/29584ed6d68a06fdf91440a018f6bc83a44fd177))



## Example
🍒 Cherry-picked example to show a long review. **Bold text** are the extracted **rationales**.   Korean is the original review. English version is first machine translated by Naver Papago, and then rationales are highlighted by myself.

---
*(Top-k;  negative)*   
SNS에서 인기가 제일 많아서 처음 **샤로수길** 방문하면서 식사를 했습니다. **웨이팅에** 이름 올리고 40분 정도 기다려서 **식사했습니다.** 처음 비주얼은 와 처음 몇 숟가락은 외에의 **맛인데... 중간부터는** 속이 **느글거려** 먹는 걸 **즐기지 못했습니다.** 그냥 한번 이런 맛이 있구나 정도지 두 번은 가고 싶지 **않습니다.** 물론 개인 취향은 있겠지만, 같이 동행한 중학생과 초등학생 딸이 **맛이 없었다고 하네요.** 먼 길 가서 기대한 점심 **식사인데...** 소셜은 역시 걸러서 들어야 하나 봅니다.

*(Contiguous;  negative)*   
SNS에서 인기가 제일 많아서 처음 샤로수길 방문하면서 식사를 했습니다. 웨이팅에 이름 올리고 40분 정도 기다려서 식사했습니다. 처음 비주얼은 와 처음 몇 숟가락은 **외에의 맛인데... 중간부터는 속이 느글거려 먹는 걸 즐기지 못했습니다. 그냥 한번 이런 맛이** 있구나 정도지 두 번은 가고 싶지 않습니다. 물론 개인 취향은 있겠지만, 같이 동행한 중학생과 초등학생 딸이 맛이 없었다고 하네요. 먼 길 가서 기대한 점심 식사인데... 소셜은 역시 걸러서 들어야 하나 봅니다.

---
*(Top-k;  negative)*   
It was the most popular on SNS, so I had a meal while visiting **Sharosu-gil** for the first time. I put my name on the **waiting list** and waited for about 40 minutes to eat. Wow, the first few spoons **taste** like something else. I **couldn't enjoy eating** because I **felt sick from the middle.** I **don't** want to go there twice. Of course, there will be personal preferences, but the middle school and elementary school daughter who accompanied them **said it was not delicious.** It's a long-awaited **lunch**. I think you should listen to social media.

*(Contiguous;  negative)*   
It was the most popular on SNS, so I had a meal while visiting Sharosu-gil for the first time. I put my name on the waiting list and waited for about 40 minutes to eat. Wow, the first few spoons **taste like something else. I couldn't enjoy eating because I felt sick from the middle.** I don't want to go there twice. Of course, there will be personal preferences, but the middle school and elementary school daughter who accompanied them said it was not delicious. It's a long-awaited lunch. I think you should listen to social media.


## Acknowledgements

This work was done in the class, *Data Science Capstone Project* at Seoul National University. I thank professor [Wen-Syan Li](https://sites.google.com/view/appliedmllab/home) for the support and [Seoul National University Graduate School of Data Science](https://gsds.snu.ac.kr/) for the compute resources.
