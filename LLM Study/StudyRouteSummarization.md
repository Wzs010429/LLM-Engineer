# LLM 入门路线汇总

这个md文件主要包含了对于小白入门LLM的相关书籍/Paper/课程/GitHub Repo的推荐，以及从0开始的技术路线梳理。



## 数学基础

这部分选择性补充复习，应该大家都会有一些基础，但是关键的定理之类的想不起来了就专门去搜就好了

放几个我觉得比较好的吴恩达的网课和速查知识点链接

- [线性代数](https://www.bilibili.com/video/BV1Pg4y1X7Pa/?spm_id_from=333.999.0.0%5C&vd_source=85caee005d65cf61873d68b8ae05d319)
- [概率论](https://www.bilibili.com/video/BV1WH4y1q7o6/?vd_source=a6ba568092f18e40c35d199a6981f8dc)
- [数学基础速查](https://github.com/ben1234560/AiLearning-Theory-Applying/blob/master/%E5%BF%85%E5%A4%87%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80.md)
- [图解AI数学基础](https://www.showmeai.tech/tutorials/83)
- [机器学习知识点速查](http://www.ai-start.com/CS229/2.CS229-Prob.html)

理论上掌握下面的全部内容，数学基本够用了：

- 线性代数：矩阵运算，范数，特征分解，SVD分解，逆矩阵，常用距离度量
- 概率论：矩阵运算，范数，特征分解，SVD分解，逆矩阵，常用距离度量
- 微积分：求导/链式法则，最优化理论，梯度下降，牛顿法，雅可比矩阵，Hessian矩阵
- 信息论：墒，交叉墒，互信息，最大熵


## 书籍推荐

首先推荐一个电子书收集巨全的[Github Repo](https://github.com/ytin16/awesome-machine-learning-1)，几乎所有的AI专业书都可以找得到，包括一会在下面提到的电子书link这里面都会有

### 深度学习入门

- [《动手学深度学习》李沐](https://zh.d2l.ai/)，沐神入门深度学习的好书，主要看pytorch版本的就可以了，mxnet用的比较少，B站也有配套的网课，沐神亲自讲的，也比较通俗易懂，[链接在这](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497)
- [《深度学习》花书](https://github.com/ytin16/awesome-machine-learning-1/blob/master/Deep-Learning%E8%8A%B1%E4%B9%A6-%E3%80%8A%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E3%80%8B-%E4%B8%AD%E6%96%87%E7%89%88.pdf)，这部分理论比较多，可以和李沐的《动手学深度学习》二选一即可

### 大模型LLM入门


书还是比较少的，一般都是直接从Papers或者博客来学习新知识

- 一本可以当作闲书来看的《导论》：[大语言模型](https://github.com/LLMBook-zh/LLMBook-zh.github.io)，这本书讲的不算特别深入而且比较新（24年4月），可以对LLM的发展和关键的前沿技术有比较直观的了解（虽然有400页但是不要被吓到，干货不算特别多，主要是概念）


## Papers

- [Attention is all you need](https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf) 注意力机制开山论文
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 提出了BERT模型，通过双向Transformer进行语言模型预训练，改变了NLP的许多基准任务
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) OpenAI提出的GPT-3模型，展示了大规模模型在Few-Shot和Zero-Shot学习中的强大能力
- [Word2Vec: Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) 基于词向量的生成模型，通过CBOW和Skip-gram方法学习词的分布式表示
- [Llama: Meta AI's Large Language Model](https://arxiv.org/abs/2302.13971) Llama模型，Meta AI发布的大型语言模型
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) LoRA技术，一种对大型语言模型进行有效微调的方法
- [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) CoT技术，通过链式思考提示来引导大型语言模型进行推理
- [A Survey on LoRA of Large Language Models](https://arxiv.org/pdf/2407.11046)关于轻量化微调的survey
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) RAG的技术汇总的survey
- [Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey](https://arxiv.org/abs/2403.14608) 轻量化微调的Survey

一些survey可能会比较劝退（太长，一般读不动），可以用ChatGPT或者Kimi协助阅读

## 课程

- [斯坦福 CS224n](https://web.stanford.edu/class/cs224n/)，nlp经典课程，前面的链接是对应斯坦福官网放出的最新ppt，网课见B站[链接](https://www.bilibili.com/video/BV1d6421f7oW/?vd_source=a6ba568092f18e40c35d199a6981f8dc)，视频内容可能和ppt有小幅度不匹配，但是大致内容基本一致
- [DataWhale 大模型工程入门](https://space.bilibili.com/431850986/channel/collectiondetail?sid=3306540)，这是一个面向工程开发的LLM入门理论课程系列，选看，后面应用开发可以不看，前面的入门介绍比较不错
- [DataWhale Prompt 工程师认证](https://datawhaler.feishu.cn/wiki/BhVQw3FlFitUTAkS4oXcSyJanph) 这个对于提示工程的理论介绍比较详细，学完了还能拿个证


## GitHub Repo

- 这个[Issue](https://github.com/ninehills/blog/issues/97)提到了LLM工程师的入门路线，也比较有参考价值
- 深度学习（花书），[源码复现（pytorch）](https://github.com/MingchaoZhu/DeepLearning)
- 动手学深度学习，[源码复现（pytorch）](https://github.com/ShusenTang/Dive-into-DL-PyTorch)
- [大模型部署指南](https://github.com/datawhalechina/self-llm)，还在更新的
- [技术博客](https://mlabonne.github.io/blog/) 就是一些技术博文 关于Llama 3微调这种，对应的github Repo对LLM入门学习路线也有比较清晰的讲解，[链接在这](https://github.com/mlabonne/llm-course)


## 技术路线

这里分享一个DataWhale推荐的大模型入门路线：[链接](https://datawhaler.feishu.cn/wiki/X9AVwtmvyi87bIkYpi2cNGlIn3v)，这个相对来讲几门课程比较完善，适合小白入门。

具体的NLP和LLM技术路线如下，我自己整理的，可能不算特别全，但是基本涵盖主要分支了

### 数学基础

详见上文 数学基础

### 编程基础

Python，Pytorch足以

### 自然语言处理基础

这部分相对来讲略微过时，简单了解原理即可，不需要细扣技术细节

- 文本预处理：分词、去停用词、词干化和词形还原
- 词向量表示：理解词袋模型（BoW）、TF-IDF、Word2Vec、GloVe等
- 常见NLP任务：
    - 文本分类：Spam Detection、情感分析
    - 文本生成：语言模型（如n-gram、基于RNN的简单生成模型）
    - 序列标注：命名实体识别（NER）、词性标注（POS tagging）

下面这部分十分重要：

- 序列到序列模型：RNN、LSTM、GRU的基本概念与实现
- 注意力机制：理解注意力的作用，学习Self-Attention、Multi-Head Attention
- Transformer：重点学习Transformer架构（如编码器、解码器、自注意力机制），并了解其如何用于机器翻译等任务
- BERT与GPT：理解预训练语言模型的概念，深入理解BERT和GPT的工作原理及在迁移学习中的应用


### 大模型入门

- ChatGPT，Llama 2/3，GPT-4 技术报告，了解他们是如何在Transformer架构上进行变种
- Prompt Engineering，提示工程
- PEFT，轻量化微调，大模型随着参数量的增加微调变得不那么现实，费时费力费钱，LoRA，QLoRA为代表性的轻量化微调开始展现优势
- RAG，检索增强生成


目前到这里足够了，后续都是工程相关，对技术研究帮助不是特别大

## Others

网上开源资料很多很杂，尤其是B站一堆卖课录屏那种视频完全不建议去看，讲的很浅而且漏洞百出，有针对性针对技术细节去寻找资料比全程跟一个系列视频要高效很多

推荐[DataWhale](https://www.datawhale.cn/)，这个开源组织的系列课程都很不错，并且活跃度很高，定期它们会组织“组队学习”活动，对初学者有系统学习路线，遇到感兴趣的topic强烈建议几个小伙伴一起去学，[GitHub Repo](https://github.com/datawhalechina)

NLP基础知识，需要有，但是不是必需，理论上具备一些数学知识就可以直接上手大模型