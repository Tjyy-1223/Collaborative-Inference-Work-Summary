## DNN云边协同工作汇总(持续更新)

云边协同旨在充分利用云边端资源完成DNN任务的推理计算，主要可以分为三个方面

+ 垂直划分：将整体模型进行划分，分别放置在云边端上进行推理。
+ 水平划分：将模型的某一层分为多个部分，利用多个边缘节点并行计算。
+ 资源划分：在云边端资源上利用各种策略进行资源调度和分配。

下面分别从三个方面汇总相关的论文工作。



### 1 垂直划分

将整体模型进行划分，分别放置在云边端上进行推理。

#### 1.1 链式拓扑

垂直划分首次由neurosurgeon这篇论文提出，首次提出了云边协同+模型划分的过程来降低模型推理时延。

+ [Collaborative Intelligence Between the Cloud and Mobile Edge](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2019_2020/papers/kang_asplos_2017.pdf)（Neurosurgeon）
  + 2017
  + 出自期刊 ASPLOS 级别 CCF-A
  + 智能划分、云边协同
+ [Context-aware Adaptive Surgery- A Fast and Effective Framework](https://dl.acm.org/doi/abs/10.1145/3478073)（CAS）
  + 2021
  + 提出了基于Kd树最近邻搜索的GADS算法，作为CAS的核心。
+ [Enabling Cooperative Inference of Deep Learning on Wearables and Smartphones](https://www.semanticscholar.org/paper/Enabling-Cooperative-Inference-of-Deep-Learning-on-Xu-Qian/c59f8f54c8caf420529afb9ddd875153f34c4280) （CoINF）
  + 2017
  + 预测模型时延构建，有参考价值，写的比较详细；安卓系统实现原型 + 可穿戴设备。
+ [IONN- Incremental Offloading of Neural Network Computations from Mobile Devices to Edge Servers](https://dl.acm.org/doi/10.1145/3267809.3267828)（IONN）
  + 2018
  + 出自SoCC 级别为CCF-B
  + 关注了模型传输，对DNN模型进行分区并增量上传-协同执行。
+ [An adaptive DNN inference acceleration framework with end–edge–cloud collaborative computing](https://www.sciencedirect.com/science/article/abs/pii/S0167739X22003570) （ADAF）
  + 2023
  + 云-边-端联合协同计算。

#### 1.2 DAG拓扑

DADS使用图论中的最大流最小割算法对DAG拓扑结构进行了分析，解决了一部分含有拓扑结构的模型的划分问题。

+ [Dynamic Adaptive DNN Surgery for Inference Acceleration on the Edge](https://ieeexplore.ieee.org/document/8737614)（DADS）
  + 2019
  + 出自INFOCOM，级别为CCF-A
  + 在边缘设备和云服务器之间划分，用图论解决resnet等非链式结构。
+ [DNN Real-Time Collaborative Inference Acceleration with Mobile Edge Computing](https://ieeexplore.ieee.org/document/9892582)（CIC）
  + 2022
  + 出自IJCNN，级别为CCF-C
  + 模型压缩+优化方法来提高模型划分的效率，减少模型划分的决策时间。
+ [Mobility-Included DNN Partition Offloading from Mobile Devices to Edge Clouds](https://www.mdpi.com/1424-8220/21/1/229)（MDPO）
  + 2021
  + 出一种包含移动性的DNN分区卸载算法(MDPO)，最小化完成DNN作业的总延迟。

#### 1.3 结合提前退出机制

+ [BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks](https://arxiv.org/abs/1709.01686)
  + 讲解了DNN模型的提前退出机制

+ [Edge Intelligence: On-Demand Deep Learning Model Co-Inference with Device-Edge Synergy](https://arxiv.org/abs/1806.07840)（Edgent）
  + 2018
  + 出自SIGCOMM，级别为CCF-A
  + 使用提前退出机制，降低DNN云边协同推理的端到端时延。
+ [Edge AI: On-Demand Accelerating Deep Neural Network Inference via Edge Computing](https://arxiv.org/abs/1910.05316)（Edgent）
  + 2019
  + 出自TWC，级别为CCF-B
  + 在2018版本上加入了对于动态网络的适应性
+ [On-demand inference acceleration for directed acyclic graph neural networks over edge-cloud collaboration](https://www.sciencedirect.com/science/article/abs/pii/S0743731522001964)（EDDI）
  + 2023
  + 出自JPDC，级别为CCF-B
  + 可以认为是结合了DADS + BranchyNet算法，使算法能够适应DAG拓扑结构。

#### 1.4 predictor预测器构建

对于DNN模型推理时延的预测是垂直划分中重要的一部分，总结了一些讲解推理时延预测的论文，如下。

+ [inference latency prediction at the edge](https://arxiv.org/pdf/2210.02620.pdf)
+ [nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices](https://air.tsinghua.edu.cn/pdf/nn-Meter-Towards-Accurate-Latency-Prediction-of-Deep-Learning-Model-Inference-on-Diverse-Edge-Devices.pdf)
+ [Pruning convolutional neural networks for resource efficient inference](https://openreview.net/pdf?id=SJGCiw5gl)
+ [PALEO: A performance model for deep neural networks](https://openreview.net/pdf?id=SyVVJ85lg)
+ [Predicting Latency of Neural Network Inference](http://cs230.stanford.edu/projects_fall_2020/reports/55793069.pdf)



### 2 水平划分

对DNN中的某一层进行分段划分，或者像网格一样划分后，使用多个边缘设备并行计算。

+ [Modnn: Local distributed mobile computing system for deep neural network](https://ieeexplore.ieee.org/document/7927211)（MoDNN）
  + 2017
  + 出自DATE，级别为CCF-B
  + 分布式+CNN逐层并行计算
+ [Adaptive parallel execution of deep neural networks on heterogeneous edge device](https://dl.acm.org/doi/10.1145/3318216.3363312)（AOFL）
  + 2019
  + 出自SEC，级别为CCF-A
  + 分布式+CNN逐层并行计算
+ [Distributed inference acceleration with adaptive DNN partitioning and offloading](https://ieeexplore.ieee.org/document/9155237)（DINA）
  + 2020
  + 出自INFOCOM，级别为CCF-A
  + 根据每个ES的计算资源和可用的ES的数量灵活的划分输入张量；沿最大边进行划分；使用匹配理论解决问题。
+ [DeepSlicing: COllaborative and adaptive CNN inference with low latency](https://ieeexplore.ieee.org/document/9353250)
  + 2021
  + 出自TPDS，级别为CCF-A
  + 分布式 + fusion block 融合块并行计算。
+ [DeepThings: Distributed adaptive deep learning inference on resource-constrained IoT edge clusters](https://ieeexplore.ieee.org/document/8493499)
  + 2018
  + 出自TCAD，级别为CCF-B
  + DeepThings中的单个任务涉及整个CNN，导致重叠计算和冗余任务；通过划分DNN layer 实现并行加速。
+ [Collaborative edge computing for distributed CNN inference acceleration using receptive field-based segmentation](https://www.sciencedirect.com/science/article/pii/S1389128622002638)
  + 2022
  + 出自CN，级别为CCF-B
  + 使用动态规划解决融合块的选取问题；使用贪心策略实现边缘设备的挑选。



### 3 划分调度

在多个边缘服务器和边缘设备的集群中，使DNN任务进行合理调度，降低任务完成的平均时延或平均能耗。在进行调度的过程中可以使用垂直划分和水平划分，也可以直接将一个DNN任务作为划分单位。

可以使用传统算法进行调度，也可以利用强化学习进行调度。

#### 3.1 传统方法

+ [MODI: Mobile Deep Inference Made Efficient by Edge Computing](https://www.usenix.org/system/files/conference/hotedge18/hotedge18-papers-ogden.pdf)（MODI）
  + 2018
  + 提出了可行方案，但没有具体研究： 1) 运行时动态选择最佳模型。 2) 在边缘服务器上存储高质量的模型 3) 定期在边缘位置更新模型，保证低时延。

+ [Fine-grained Cloud Edge Collaborative Dynamic Task Scheduling Based on DNN Layer-Partitioning](https://www.computer.org/csdl/proceedings-article/msn/2022/645700a155/1LUtVyqXmdW)（DLPDTS）
  + 2022
  + 出自MSN，级别为CCF-C
  + 提出了一种基于DLP的云边协同细粒度动态任务调度机制，该机制包括DLP选点算法和动态任务调度算法两部分。

#### 3.2 强化学习

+ [“DRL + FL”: An intelligent resource allocation model based on deep reinforcement learning for Mobile Edge Computing](https://www.sciencedirect.com/science/article/abs/pii/S014036641932122X)
  + 2020
  + 本篇论文的关注点在于强化学习解决资源分配，但是其主要解决的是使用联邦学习优化强化学习中agent的训练。

+ [Accuracy-Guaranteed Collaborative DNN Inference in Industrial IoT via Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9170818)
  + 2020 TII - CCFC
  + 主要考虑了采样率自适应问题（调整输入数据）。 1）将问题表述为约束马尔可夫决策过程(CMDP)，综合考虑了推理任务卸载和边缘计算资源分配。 2）通过一般强化学习(RL)算法直接求解。
  + 虽然是应用于工业物联网场景，但是对于模型调度和计算资源分配还是一篇非常值得读的论文，以及其中如何用强化学习构建场景的过程。
