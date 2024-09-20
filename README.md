# gpt系列

#### 介绍
gpt原理及其使用

1. [【GPT系列】什么是Milvus向量数据库？定义概念原理示例索引](https://zhuanlan.zhihu.com/p/692395994)
2. [【langchain示例系列】与RAG的问答（Q&A with RAG）](https://zhuanlan.zhihu.com/p/700383948)
3. [【GPT系列】阿里云大模型快速入门体验](https://zhuanlan.zhihu.com/p/692871932)
4. [【gpt系列】国内头部gpt工具Kimi使用手册主要概念](https://zhuanlan.zhihu.com/p/719840595)
5. [【gpt系列】OpenAI Python API库from openai import OpenAI用法示例拓展详细说明](https://zhuanlan.zhihu.com/p/719868531)
6. [【gpt系列】如何使用 Kimi API示例演示详解](https://zhuanlan.zhihu.com/p/719897820)
7. [人工智能数学基础目录专栏](https://zhuanlan.zhihu.com/p/692873436)


![img](https://pic2.zhimg.com/80/v2-0c8a0369fabf62a78ecd8a449bdd3c2d_720w.webp)

## 【GPT系列】什么是Milvus[向量数据库](https://zhida.zhihu.com/search?q=向量数据库&zhida_source=entity&is_preview=1)？

*源自专栏《[docker常用命令系列&&k8s系列目录导航](https://zhuanlan.zhihu.com/p/680652806)》*

## Milvus

Milvus是在2019年创建的，其唯一目标是**存储、索引和管理**由[深度神经网络](https://zhida.zhihu.com/search?q=深度神经网络&zhida_source=entity&is_preview=1)和其他机器学习（ML）模型生成的**大规模[嵌入向量](https://link.zhihu.com/?target=https%3A//www.milvus-io.com/overview%23Embedding-vectors)。**

作为一个专门设计用于处理输入向量查询的数据库，它能够处理万亿级别的向量索引。与现有的[关系型数据库](https://zhida.zhihu.com/search?q=关系型数据库&zhida_source=entity&is_preview=1)主要处理遵循预定义模式的结构化数据不同，Milvus从底层设计用于处理从[非结构化数据](https://link.zhihu.com/?target=https%3A//www.milvus-io.com/overview%23Unstructured-data)转换而来的嵌入向量。

随着互联网的发展和演变，非结构化数据变得越来越常见，包括电子邮件、论文、[物联网传感器](https://zhida.zhihu.com/search?q=物联网传感器&zhida_source=entity&is_preview=1)数据、Facebook照片、[蛋白质结构](https://zhida.zhihu.com/search?q=蛋白质结构&zhida_source=entity&is_preview=1)等等。为了使计算机能够理解和处理非结构化数据，使用嵌入技术将它们转换为向量。Milvus存储和索引这些向量。Milvus能够通过计算它们的相似距离来分析两个向量之间的相关性。如果两个嵌入向量非常相似，则意味着原始数据源也很相似。



![img](https://picx.zhimg.com/80/v2-9c44d2d8b9b78721c69ff84318a3828b_720w.webp)

Workflow

[(opens in a new tab)](https://link.zhihu.com/?target=https%3A//milvus.io/static/3b65292e6a7d800168c56ecfd8f7109e/1b5bd/milvus_workflow.jpg)

Milvus workflow.

## 关键概念

如果您对向量数据库和相似度搜索的世界还不熟悉，请阅读以下关键概念的解释，以更好地理解。

了解更多关于[Milvus词汇表](https://link.zhihu.com/?target=https%3A//www.milvus-io.com/glossary)。

### 非结构化数据

非结构化数据包括图像、视频、音频和自然语言等信息，这些信息不遵循预定义的模型或组织方式。这种数据类型占据了世界数据的约80%，可以使用各种人工智能（AI）和机器学习（ML）模型将其转换为向量。

### 嵌入向量

嵌入向量是对非结构化数据（如电子邮件、物联网传感器数据、Instagram照片、蛋白质结构等）的特征抽象。数学上，嵌入向量是一个浮点数或[二进制数](https://zhida.zhihu.com/search?q=二进制数&zhida_source=entity&is_preview=1)的数组。现代的嵌入技术被用于将非结构化数据转换为嵌入向量。

### 向量相似度搜索

向量相似度搜索是将向量与数据库进行比较，以找到与查询向量最相似的向量的过程。使用近似最近邻搜索算法加速搜索过程。如果两个嵌入向量非常相似，那么原始数据源也是相似的。

## 原理

Milvus 是一款[云原生](https://zhida.zhihu.com/search?q=云原生&zhida_source=entity&is_preview=1)向量数据库，其设计原理基于分离存储与计算，以提高弹性、灵活性和性能。

以下是 Milvus 的设计原理概述：

1. **分离存储与计算：** Milvus 将存储和计算分离，通过独立的存储层和计算层实现数据的高效管理和处理。
2. **无状态组件：** Milvus 中的所有组件都是无状态的，这意味着系统的各个部分可以独立运行，降低了组件之间的耦合度，提高了系统的可扩展性和灵活性。
3. **[系统架构](https://zhida.zhihu.com/search?q=系统架构&zhida_source=entity&is_preview=1)层次：** Milvus 的系统架构分为四个层次，包括访问层、协调器服务、工作节点和存储层，每个层次承担不同的任务和功能。

- **访问层：** 由一组无状态代理组成，作为系统的前端和用户端点，负责处理用户的请求和交互。
- **协调器服务：** 充当系统的大脑，负责[任务调度](https://zhida.zhihu.com/search?q=任务调度&zhida_source=entity&is_preview=1)和指令分配，将任务分配给工作节点进行执行。
- **工作节点：** 执行来自协调器服务的指令，是系统的执行者，负责执行用户触发的数据操作命令（DML/DDL）。
- **存储：** 包括元数据存储、日志代理和[对象存储](https://zhida.zhihu.com/search?q=对象存储&zhida_source=entity&is_preview=1)等组件，负责数据的持久化和管理，是系统的数据骨架。

1. **高效数据管理：** Milvus 的设计旨在实现高效的向量数据管理和查询，通过[分层架构](https://zhida.zhihu.com/search?q=分层架构&zhida_source=entity&is_preview=1)和组件设计，提高了系统的性能和可靠性。

Milvus 的设计原理注重分层架构、组件化设计和无状态服务，以实现高效的向量[数据处理](https://zhida.zhihu.com/search?q=数据处理&zhida_source=entity&is_preview=1)和管理，在大规模向量数据集的存储、索引和查询中发挥重要作用。

## 示例应用

Milvus使得向应用中添加相似性搜索变得容易。Milvus的示例应用包括：

### 图像相似性搜索

- 使图像可搜索，并即时返回来自大型数据库中最相似的图像。
- 要构建这样的图像相似性搜索系统，
- 请下载包含20个类别的17125个图像的PASCAL VOC图像集。或者，您可以准备自己的图像数据集。使用YOLOv3进行物体检测和ResNet-50进行图像[特征提取](https://zhida.zhihu.com/search?q=特征提取&zhida_source=entity&is_preview=1)。通过这两个ML模型之后，图像被转换为256维向量。
- 然后将向量存储在Milvus中，并由Milvus**自动生成每个向量的唯一ID**。
- 然后使用MySQL将**向量ID映射到数据集中的图像**。每当您上传新图像到图像搜索系统时，它将被转换为新向量，并与以前存储在Milvus中的向量进行比较。
- 然后，Milvus返回最相似向量的ID，您可以在MySQL中查询相应的图像。



![img](https://pic4.zhimg.com/80/v2-8f812b5b05d52b1b28546c13218440df_720w.webp)

image_search

[(opens in a new tab)](https://link.zhihu.com/?target=https%3A//milvus.io/static/24846258d3594e2f90a09ad9f4c9b8c9/b87a0/image_search.png)

Workflow of a reverse image search system.



![img](https://pic1.zhimg.com/80/v2-ef56819796e8f27e0ec6ba567fe51c7e_720w.webp)

image_search_demo



### 视频相似性搜索

- 通过将关键帧转换为向量，然后将结果输入Milvus，可以在几乎实时的时间内搜索和推荐数十亿个视频。
- 人们看完喜欢的电影或视频后，可以轻松地截图并通过各种社交平台发布他们的想法。当关注者看到截图时，如果电影名称没有在帖子中明确说明，他们很难确定电影的名称。为了确定电影的名称，人们可以利用视频相似度搜索系统。通过使用该系统，用户可以上传图像并获得包含与上传图像相似的关键帧的视频或电影。
- 构建一个视频相似度搜索系统。本教程使用Tumblr上的约100个动画gif来构建系统。但是，您也可以准备自己的视频数据集。该系统首先使用OpenCV从视频中提取关键帧，然后使用ResNet-50获取每个关键帧的[特征向量](https://zhida.zhihu.com/search?q=特征向量&zhida_source=entity&is_preview=1)。所有向量都存储在Milvus中，并进行搜索，Milvus将返回相似向量的ID。然后将ID映射到存储在MySQL中的相应视频。



![img](https://picx.zhimg.com/80/v2-57dccf94c058a63ce67a7be14dd48fd1_720w.webp)

video_search



### 音频相似性搜索

- 快速查询大量[音频数据](https://zhida.zhihu.com/search?q=音频数据&zhida_source=entity&is_preview=1)，如语音、音乐、音效和表面相似的声音。
- 语音、音乐、音效等各种类型的音频搜索使得可以快速查询海量音频数据，并找到相似的声音。音频相似性搜索系统的应用包括识别相似的音效、减少[知识产权侵权](https://zhida.zhihu.com/search?q=知识产权侵权&zhida_source=entity&is_preview=1)等。音频检索可用于实时搜索和监控在线媒体，以打击知识产权侵权。它也在音频数据的分类和[统计分析](https://zhida.zhihu.com/search?q=统计分析&zhida_source=entity&is_preview=1)中扮演着重要的角色。
- 构建一个音频相似性搜索系统，可以返回相似的声音片段。上传的音频片段使用PANNs转换为向量。这些向量存储在Milvus中，Milvus会自动生成每个向量的唯一ID。然后，用户可以在Milvus中进行[向量相似度](https://zhida.zhihu.com/search?q=向量相似度&zhida_source=entity&is_preview=1)搜索，并查询与Milvus返回的唯一向量ID对应的音频片段数据路径。



![img](https://pic2.zhimg.com/80/v2-4699547e89fe3fd0922e4af79fa1e175_720w.webp)

Audio_search

[(opens in a new tab)](https://link.zhihu.com/?target=https%3A//milvus.io/static/408dacaa6e9ca67fec59564953195a01/98b00/audio_search.png)

Workflow of an audio similarity search system.



![img](https://pic4.zhimg.com/80/v2-47a00c44268a2fab27fa11ca996402c9_720w.webp)

Audio_search_demo

[(opens in a new tab)](https://link.zhihu.com/?target=https%3A//milvus.io/static/eb203dda58fca2bc1cf8a3f385063cdb/bbbf7/audio_search_demo.png)

Demo of an audio similarity search system.

### 分子相似性搜索

- 针对指定分子进行极快的相似性搜索、子结构搜索或超结构搜索。
- 药物开发是新药研发的重要组成部分。药物开发过程包括靶点选择和确认。当发现片段或引导化合物时，研究人员通常会在内部或商业化合物库中搜索类似化合物，以发现结构活性关系（SAR）、化合物可用性。最终，他们将评估引导化合物的潜力，以优化为候选化合物。为了从数十亿规模的化合物库中发现可用的化合物，通常检索化学指纹以进行亚结构搜索和分子相似性搜索。
- 构建一种分子相似性搜索系统，该系统可以检索特定分子的亚结构、超结构和类似结构。RDKit是一款开源的化学信息学软件，可以将分子结构转换为向量。然后，这些向量存储在Milvus中，Milvus可以对向量进行相似性搜索。Milvus还会自动为每个向量生成唯一的ID。向量ID和分子结构的映射存储在MySQL中。



![img](https://pic1.zhimg.com/80/v2-e8ed06bf66b0247cb1dd9a47a03134e8_720w.webp)

molecular

[(opens in a new tab)](https://link.zhihu.com/?target=https%3A//milvus.io/static/b3bac1ee6c20ddd2a8af37cb9479f66e/7e99c/molecular.png)

Workflow of a molecular [similarity](https://zhida.zhihu.com/search?q=similarity&zhida_source=entity&is_preview=1) search system.



![img](https://pic4.zhimg.com/80/v2-7c887b668453d20f1359d38027f28ce3_720w.webp)

molecular

[(opens in a new tab)](https://link.zhihu.com/?target=https%3A//milvus.io/static/70c9b0d2fa5d883b1de4ad0625dc06aa/40619/molecular_demo.jpg)

Demo of a molded similarity search system.

### [推荐系统](https://zhida.zhihu.com/search?q=推荐系统&zhida_source=entity&is_preview=1)

- 根据用户行为和需求推荐信息或产品。
- 推荐系统是信息过滤系统的一个子集，可用于个性化电影、音乐、产品和信息流推荐等各种场景。与[搜索引擎](https://zhida.zhihu.com/search?q=搜索引擎&zhida_source=entity&is_preview=1)不同，推荐系统不需要用户准确描述其需求，而是通过分析用户行为来发现用户的需求和兴趣。
- 如何构建一个电影推荐系统，该系统可以建议符合用户兴趣的电影。要构建这样的推荐系统，首先需要下载一个与电影相关的数据集。本教程使用MovieLens 1M。或者，您可以准备自己的数据集，其中应包括用户对电影的评分、用户的人口统计特征和电影描述等信息。使用PaddlePaddle将用户ID和特征组合并将其转换为256维向量。以类似的方式将电影ID和特征转换为向量。将电影向量存储在Milvus中，并使用用户向量进行相似性搜索。如果用户向量与电影向量相似，Milvus将返回电影向量和其ID作为推荐结果。然后使用存储在Redis或MySQL中的电影向量ID查询电影信息。



![img](https://picx.zhimg.com/80/v2-4c1784d5040d9c74c49b637b073ba77f_720w.webp)

recommender_system

[(opens in a new tab)](https://link.zhihu.com/?target=https%3A//milvus.io/static/46df1b9dcc35b52455c5e9ede92ca5ae/a27c6/recommendation_system.png)

Workflow of a recommender system.

### 问答系统

- 交互式数字问答聊天机器人，自动回答用户的问题。
- 问答系统是自然语言处理领域中常见的实际应用，典型的QA系统包括[在线客服](https://zhida.zhihu.com/search?q=在线客服&zhida_source=entity&is_preview=1)系统、QA聊天机器人等等。大多数问答系统可以归类为：生成式或检索式，单轮或多轮，开放域或特定领域问答系统。
- 构建一个QA系统，该系统可以将新用户的问题链接到先前存储在向量数据库中的大量答案。为构建这样的聊天机器人，准备自己的问题和相应答案的数据集。将这些问题和答案存储在关系型数据库MySQL中。然后使用自然语言处理（NLP）的机器学习（ML）模型BERT将问题转换为向量。这些问题向量被存储并索引在Milvus中。当用户输入新问题时，BERT模型也将其转换为一个向量，Milvus会搜索与这个新向量最相似的问题向量。QA系统将返回最相似问题的相应答案。



![img](https://pic3.zhimg.com/80/v2-4ff673d3d78701e739a59e0b370d50e0_720w.webp)

QA_Chatbot



### DNA序列分类

- 通过比较相似的DNA序列，在毫秒级别准确地分类一个基因。
- DNA序列是基因可追溯性、物种识别、疾病诊断等许多领域的流行概念。而所有行业都渴望更智能、更高效的[研究方法](https://zhida.zhihu.com/search?q=研究方法&zhida_source=entity&is_preview=1)，特别是生物和医学领域，人工智能引起了很大关注。越来越多的科学家和研究人员正在为生物信息学领域的机器学习和深度学习做出贡献。为了使实验结果更加令人信服，一种常见的选择是增加样本量。与基因组学的大数据协作带来了更多的应用可能性。然而，传统的序列比对具有局限性，使其不适用于大型数据集。为了在现实中做出更少的权衡，向量化是大型DNA序列数据集的一个很好的选择。
- 构建DNA序列分类模型。本教程使用CountVectorizer提取DNA序列的特征并将其转换为向量。然后，这些向量存储在Milvus中，它们对应的DNA分类存储在MySQL中。用户可以在Milvus中进行向量相似性搜索，并从MySQL中检索相应的DNA分类。



![img](https://pic4.zhimg.com/80/v2-d6f286ac86c073a1dfe82e2cd6627957_720w.webp)

dna

[(opens in a new tab)](https://link.zhihu.com/?target=https%3A//milvus.io/static/2ea19b2d3f624295527d3a0d330d808e/c9fe3/dna.png)

Workflow of a DNA sequence classification model.

### 文本搜索引擎

- 通过将关键字与[文本数据库](https://zhida.zhihu.com/search?q=文本数据库&zhida_source=entity&is_preview=1)进行比较，帮助用户找到他们正在寻找的信息。
- 在自然语言处理（NLP）领域中，Milvus的一个主要应用是文本搜索引擎。它是一个强大的工具，可以帮助用户找到他们正在寻找的信息。甚至可以呈现出难以找到的信息。文本搜索引擎会将用户输入的关键字或语义与文本数据库进行比较，然后返回符合特定标准的结果。
- 构建一个文本搜索引擎。本教程使用BERT将文本转换为固定长度的向量。Milvus用作存储和向量相似性搜索的向量数据库。然后使用MySQL将Milvus生成的向量ID映射到文本数据。



![img](https://pic3.zhimg.com/80/v2-6ff68903bd7a6e8088519cba3bf0adc6_720w.webp)

text_search_engine



## 向量数据库索引

向量数据库索引是针对向量数据进行检索和查询的数据库索引技术，主要用于存储和高效查询大规模向量数据集，如图片特征向量、文本向量等。向量数据库索引通常基于向量相似度的计算，用于快速搜索和检索相似的向量数据。

### 主要特点

- **基于向量相似度：** 向量数据库索引根据向量之间的相似度进行索引和检索，而非传统的键值对索引。
- **高效查询：** 通过利用向量相似度计算算法，实现高效的向量[数据检索](https://zhida.zhihu.com/search?q=数据检索&zhida_source=entity&is_preview=1)和查询。
- **支持多维向量：** 向量数据库索引适用于多维向量数据，如图像、音频、文本等。
- **实时性能：** 针对大规模向量数据集，向量数据库索引能够在实时或近实时情况下快速响应查询请求。

### 应用场景

- **图像搜索：** 用于图像相似性搜索和检索，可以快速找到与查询图像相似的图像数据。
- **推荐系统：** 在[个性化推荐系统](https://zhida.zhihu.com/search?q=个性化推荐系统&zhida_source=entity&is_preview=1)中，利用向量数据库索引实现用户向量和物品向量之间的相似度计算。
- **文本检索：** 用于文本相似性匹配和搜索，帮助用户查找与输入文本相似的文档或文章。
- **医疗影像分析：** 用于医疗影像数据的相似性分析和检索，帮助医生快速找到相关病例。

### 常见技术

- **LSH（Locality Sensitive Hashing）：** 一种常用的向量索引技术，用于将相似的向量映射到相同的哈希桶中，加速相似向量的搜索。
- **FAISS（Facebook AI Similarity Search）：** 由 Facebook 开发的高性能相似性搜索库，用于快速搜索高维向量数据。
- **Annoy：** 一种 C++ 库，用于构建大规模向量索引，支持近似最近邻搜索。

### 优势

- **高效性能：** 向量数据库索引能够快速高效地搜索和检索大规模向量数据。
- **准确性：** 基于向量相似度的索引技术可以精确地找到相似的向量数据。
- **适用性广泛：** 向量数据库索引适用于多种向量数据类型，可应用于不同领域的数据分析和搜索任务。

向量数据库索引是处理大规模向量数据的关键技术，能够帮助实现高效的相似性搜索和数据检索功能。

## Milvus支持的向量索引类型及特点

### 1. **FLAT：**

- **特点：** 适合小规模、百万级数据集，追求完全准确和精确的搜索结果。
- **应用场景：** 需要精确结果的场景，数据集规模相对较小的情况。

### 2. **IVF_FLAT：**

- **特点：** 量化索引，平衡精度和查询速度的理想选择。
- **应用场景：** 追求精度和查询速度平衡的场景，适用于中等规模的数据集。

### 3. **IVF_SQ8：**

- **特点：** 量化索引，降低资源消耗，适用于资源有限的场景。
- **应用场景：** 资源有限，需要减少内存和磁盘消耗的场景。

### 4. **IVF_PQ：**

- **特点：** 量化索引，追求高查询速度，可牺牲一定精度。
- **应用场景：** 对查询速度要求高，能够接受一定精度损失的场景。

### 5. **HNSW：**

- **特点：** 基于图形的索引，适合对搜索效率有高需求的场景。
- **应用场景：** 需要高效率搜索和查询的场景，强调搜索速度和效率。

### 6. **ANNOY：**

- **特点：** 基于树形结构的索引，适合寻求高[召回率](https://zhida.zhihu.com/search?q=召回率&zhida_source=entity&is_preview=1)的场景。
- **应用场景：** 需要高召回率和较快搜索速度的场景，适用于大规模数据集。

Milvus 支持多种向量索引类型，每种类型都有其独特的特点和适用场景，可根据具体需求选择合适的索引类型来平衡搜索速度、精度和资源消耗。

## 相似度度量方法

在 Milvus 中，相似度度量方法用于衡量向量之间的相似性，对于不同类型的数据和应用场景，选择合适的相似度度量方法可以提高分类和聚类的性能。以下是常见的相似度度量方法：

### 浮点嵌入（Floating Point Embeddings）：

1. **[欧氏距离](https://zhida.zhihu.com/search?q=欧氏距离&zhida_source=entity&is_preview=1)（L2 Distance）：**

- 适用领域： 计算机视觉（Computer Vision）领域。
- 描述： 衡量向量之间的直线距离，通常用于浮点数向量的相似度计算。

1. **内积（Inner Product）：**

- 适用领域： 自然语言处理（Natural Language Processing）领域。
- 描述： 衡量向量之间的内积，用于评估向量之间的相似性。

### 二进制嵌入（Binary Embeddings）：

1. **海明距离（Hamming Distance）：**

- **适用领域：** 自然语言处理（Natural Language Processing）领域。
- **描述：** 衡量二进制向量之间的不同位数，用于度量二进制向量的相似性。

**1. 杰卡德距离（Jaccard Distance）：**

- **适用领域：** 分子相似性搜索（Molecular Similarity Search）领域。
- **描述：** 衡量集合之间的相似性，用于评估分子结构的相似性。

1. **塔尼莫托距离（Tanimoto Distance）：**

- **适用领域：** 分子相似性搜索（Molecular Similarity Search）领域。
- **描述：** 衡量分子结构之间的相似性，常用于化学信息检索中。

1. **超结构距离（Superstructure Distance）：**

- **适用领域：** 分子相似性搜索（Molecular Similarity Search）领域。
- **描述：** 衡量分子之间的结构相似性，用于搜索具有相似超结构的分子。

1. **亚结构距离（Substructure Distance）：**

- **适用领域：** 分子相似性搜索（Molecular Similarity Search）领域。
- **描述：** 衡量分子之间的亚结构相似性，用于搜索具有相似亚结构的分子。

选择合适的相似度度量方法可以有效地衡量向量或数据之间的相似性，提高数据处理和检索的效率和准确性。

## [参考链接](https://link.zhihu.com/?target=https%3A//www.milvus-io.com/text_search_engine)
