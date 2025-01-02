## DNN云边协同工作汇总

云边协同旨在充分利用云边端资源完成DNN任务的推理计算，将整体模型进行划分后，利用终端设备、边缘服务器以及云计算中心的计算资源，将DNN划分为多个部分，分别部署在不同设备上进行推理。

+ 充分利用系统中可用的计算资源
+ 降低输入数据的传输开销

## **更多资源**

**下面是一些 dblp 的关键词，直接点开可以看到最新的关于云边协同的研究工作：**

+ [collaborative edge ](https://dblp.uni-trier.de/search?q=collaborative%20edge)
+ [dnn partition](https://dblp.uni-trier.de/search?q=dnn%20partition)
+ [cloud edge dnn](https://dblp.uni-trier.de/search?q=cloud%20edge%20dnn)



## 1 DNN Partitioning

**DNN Partitioning 主要研究如何对单个DNN任务进行协同推理**

### 1.1 链式拓扑

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
+ [Self-aware collaborative edge inference with embedded devices for IIoT](Self-aware collaborative edge inference with embedded devices for IIoT)
  + 2025
  + 针对两种典型的 IIoT 场景，即突发任务和堆叠任务，分别设计了延迟感知和吞吐量感知的协作推理算法。通过联合优化分区层和协作设备选择，可以获得以最小推理延迟和最大推理吞吐量为特征的最佳推理效率。

+ [Interpretable Switching Deep Markov Model for Industrial Process Monitoring via Cloud-Edge Collaborative Framework](https://ieeexplore.ieee.org/document/10758842)
  + 2025 IEEE Transactions on Instrumentation and Measurement
  + 工业条件的频繁变化需要及时更新和重新训练现场部署的数据驱动过程监控方法，这是边缘设备有限的计算资源无法实现的任务。
  + 本文提出了一种基于可解释切换深度马尔可夫模型（ISDMM）的工业过程监控云边协作框架。ISDMM定义了代表工作条件的离散切换变量和相应的多个转换网络。转换网络根据当前工作条件同时进行训练和切换，使ISDMM能够捕获系统在不同条件下的不同动态。




### 1.2 DAG拓扑

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
+ [MODI: Mobile Deep Inference Made Efficient by Edge Computing](https://www.usenix.org/system/files/conference/hotedge18/hotedge18-papers-ogden.pdf)（MODI）
  + 2018
  + 提出了可行方案，但没有具体研究： 1) 运行时动态选择最佳模型。 2) 在边缘服务器上存储高质量的模型 3) 定期在边缘位置更新模型，保证低时延。

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

对于DNN模型推理时延的预测是模型划分中重要的一部分，总结了一些讲解推理时延预测的论文，如下：

+ [inference latency prediction at the edge](https://arxiv.org/pdf/2210.02620.pdf)
+ [nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices](https://air.tsinghua.edu.cn/pdf/nn-Meter-Towards-Accurate-Latency-Prediction-of-Deep-Learning-Model-Inference-on-Diverse-Edge-Devices.pdf)
+ [Pruning convolutional neural networks for resource efficient inference](https://openreview.net/pdf?id=SJGCiw5gl)
+ [PALEO: A performance model for deep neural networks](https://openreview.net/pdf?id=SyVVJ85lg)
+ [Predicting Latency of Neural Network Inference](http://cs230.stanford.edu/projects_fall_2020/reports/55793069.pdf)



### 1.3 水平划分

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



## 2 Task Offloaing + Traditional Method

+ [“DRL + FL”: An intelligent resource allocation model based on deep reinforcement learning for Mobile Edge Computing](https://www.sciencedirect.com/science/article/abs/pii/S014036641932122X)
  + 2020
  + 本篇论文的关注点在于使用DDQN算法与联邦学习结合解决边缘移动网络中的资源分配问题
  + 提升了系统平均服务延迟、平均能耗以及负载均衡的问题
+ [Accuracy-Guaranteed Collaborative DNN Inference in Industrial IoT via Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9170818)
  + 2020 TII - CCFC
  + 主要考虑了采样率自适应问题（调整输入数据）。 1）将问题表述为约束马尔可夫决策过程(CMDP)，综合考虑了推理任务卸载和边缘计算资源分配。 2）通过一般强化学习(RL)算法直接求解。
  + 虽然是应用于工业物联网场景，但是对于模型调度和计算资源分配还是一篇非常值得读的论文，以及其中如何用强化学习构建场景的过程。

+ [Deep Reinforcement Learning-Based Task Offloading and Resource Allocation for Industrial IoT in MEC Federation System](https://ieeexplore.ieee.org/document/10210011)
  + IEEE Access 2023
  + 联合卸载决策和资源分配问题
  + 建模称为马尔科夫决策过程，使用DDPG-PER进行解决

+ [Partition placement and resource allocation for multiple DNN-based applications in heterogeneous IoT environments](https://ieeexplore.ieee.org/document/10014999)

  + 2023 

  + 主要讲解分区放置DNN模型

+ [Deep Reinforcement Learning Based ComputationOffloading and Trajectory Planning for Multi-UAV Cooperative Target Search](https://ieeexplore.ieee.org/document/9989360)
  + 2023 IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS
  + 新兴的边缘计算技术可以通过将任务卸载到地面边缘服务器来缓解这一问题。如何评估搜索过程以做出最优卸载决策和最优飞行轨迹是基础研究的挑战。
  + 提出了一种基于深度强化学习(DRL)的多无人机协同目标搜索计算卸载决策和飞行方向选择优化方法。
+ [Deep Reinforcement Learning Based Dynamic Trajectory Control for UAV-Assisted Mobile Edge Computing](https://ieeexplore.ieee.org/document/9354996)
  + 2022 IEEE TRANSACTIONS ON MOBILE COMPUTING
  + 考虑了一个飞行移动边缘计算平台，其中UAV作为提供计算资源的设备，并使任务从UE上卸载。
  + 提出了一种基于凸优化的轨迹控制算法(CAT)，该算法采用块坐标下降法(BCD)以迭代的方式。
  + 为了在考虑环境动态的情况下进行实时决策，我们提出了一种基于深度强化学习的轨迹控制算法(RAT)。

+ [Reinforcement Learning-Based Mobile Offloading for Edge Computing Against Jamming and Interference](Reinforcement Learning-Based Mobile Offloading for Edge Computing Against Jamming and Interference)
  + 2020 SCI-I区 SRLO基准方法
  + 提出基于强化学习的边缘计算移动卸载方案，该方案利用安全强化学习来避免选择无法满足任务计算延迟要求的风险卸载策略。
  + 该方案使移动设备在不知道任务生成模型、边缘计算模型和干扰/干扰模型的情况下，可以选择边缘设备、发射功率和卸载率，以提高其效用，包括共享增益、计算延迟、能耗和卸载信号的信噪比。
  + 还设计了一种基于深度强化学习的边缘计算移动卸载，选择卸载策略，提高计算性能。
+ [CLIO: enabling automatic compilation of deep learning pipelines across IoT and cloud](https://dl.acm.org/doi/10.1145/3372224.3419215)
  + 2020 Mobicom
  + DNN模型对于低功耗加速器来说往往太大；而对于低功耗无线电来说，带宽需求往往太高。虽然在智能手机类设备的DNN模型已经做了大量的工作，但这些方法并不适用于资源受限的小型电池供电的物联网设备。
  + CLIO提出了一种新颖的方法，以适应无线动态的渐进方式在物联网设备和云之间分割机器学习模型。我们证明了该方法可以与模型压缩和自适应模型划分相结合，创建一个物联网云划分的集成系统
  + 在GAP8低功耗神经加速器上实现了CLIO，提供了每种方法表现最佳的操作机制的详尽特征，并表明CLIO可以在资源减少时实现优雅的性能下降。



## 3 DNN Partitioning + Task Offloading

在多个边缘服务器和终端设备组成的云边端系统中，使DNN任务进行合理调度，降低任务完成的平均时延或平均能耗。在进行调度的过程中可以使用垂直划分和水平划分，也可以直接将一个DNN任务作为划分单位。

+ [Fine-grained Cloud Edge Collaborative Dynamic Task Scheduling Based on DNN Layer-Partitioning](https://www.computer.org/csdl/proceedings-article/msn/2022/645700a155/1LUtVyqXmdW)（DLPDTS）
  + 2022
  + 出自MSN，级别为CCF-C
  + 提出了一种基于DLP的云边协同细粒度动态任务调度机制，该机制包括DLP选点算法和动态任务调度算法两部分。
+ [Joint optimization of DNN partition and scheduling for mobile cloud computing](https://dl.acm.org/doi/abs/10.1145/3472456.3472468)（JPS）
  + 2021
  + 出自CCFB会议 ICCP
  + 研究了如何在连续任务情况下通过二分查找寻找最优划分策略
  + 证明了连续任务场景中最优策略的选择不是一成不变的
+ [Joint Optimization of DNN Partition and Continuous Task Scheduling for Digital Twin-Aided MEC Network With Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/10070781)
  + **2023 IEEE Access** 目前找到最新的
  + RL解决信道分配和传输功率 + 数字孪生技术
  + 传统方法模型划分
+ [Dynamic resource allocation for jointing vehicle-edge deep neural network inference](https://www.sciencedirect.com/science/article/abs/pii/S1383762121001004)
  + SCI-II 区 **DNN动态分区+资源分配** 提出一种低复杂度算法解决 2021
  + 用户设备请求增多->分配的资源是动态的，所以会导致模型最优分区也是动态的，主要解决这个问题
  + 最小化所有车辆总体时延，这是np难问题
  + 建模很好 可以参考一下
+ [Joint DNN partition and resource allocation optimization for energy-constrained hierarchical edge-cloud systems](https://ieeexplore.ieee.org/document/9937150)
  + 2022 SCI-2区
  + 关注能耗优化+ 强化学习分层，RL用来选取划分点
  + 启发式算法负责云端算力分配，使用 DDPG
  + 云边协同、任务在slot内执行完成，单云中心+单边缘服务器
+ [Joint DNN partitioning and task offloading in mobile edge computing via deep reinforcement learning](https://www.researchsquare.com/article/rs-2901233/v1)
  + 2023 比较新的领域方向 Journal of cloud computing
  + 研究了DNN划分和任务卸载的能量、延迟联合优化问题
  + 使用基于PPO的DPTO解决DNN分区和任务卸载问题
  + 多个终端设备 + 单边缘服务器，将DQN DDQN 以及 PPO进行对比
+ [Reinforcement Learning Based Energy-Efficient Collaborative Inference for Mobile Edge Computing](https://ieeexplore.ieee.org/document/9984691/)
  + 2022 SCI-I区 ，多个终端设备以及多个边缘设备
  + 使用MA-DDPG 多智能体强化学习完成任务，并在真实设备上进行推理
  + 每个边缘设备使用DDPG完成两个任务：选择分区点、选择哪个边缘设备
+ [Energy-Efficient Collaborative Inference in MEC: A Multi-Agent Reinforcement Learning Based Approach](https://ieeexplore.ieee.org/document/10064441?denied=)
  + 2022 8th International Conference on Big Data Computing and Communications (BigCom)
  
  + 最优的分区点和边缘选择取决于特定深度学习架构的推理成本模型和从设备到边缘服务器的通道模型，这在实际的MEC中是具有挑战性的。

  + 提出了一种基于多智能体强化学习的MEC节能协同推理方案，根据环境条件选择深度学习模型的分区点和协同边缘服务器。
+ [Energy-Efficient Offloading for DNN-Based Smart IoT Systems in Cloud-Edge Environments](https://ieeexplore.ieee.org/document/9497712)
  + 2020 TPDS
  + 由于大规模深度神经网络的高计算成本，直接将其部署在能量受限的物联网设备中可能是不可行的。通过将计算密集型任务卸载到云或边缘，计算卸载技术为执行深度神经网络提供了一种可行的解决方案。
  + 设计了一个新的系统能耗模型，该模型考虑了所有参与服务器(来自云和边缘)和物联网设备的运行时、切换和计算能耗。
  + 提出了一种基于遗传算法算子自适应粒子群优化算法(SPSO-GA)的新型节能卸载策略，有效地对具有分层划分操作的DNN层进行卸载决策，降低了编码维数，提高了SPSO-GA的执行时间。
  
+ [Edge-Assisted Distributed DNN Collaborative Computing Approach for Mobile Web Augmented Reality in 5G Networks](https://ieeexplore.ieee.org/document/9040203)
  + 2020 IEEE Networks
  + 探索了用于云、边缘和移动 Web 浏览器之间协作的细粒度和自适应 DNN 分区。
  + 提出了一种专为边缘平台设计的差异化 DNN 计算调度方法。
  + 一方面，在不降低用户体验的情况下在移动 Web 上执行部分 DNN 计算（即保持响应延迟低于特定阈值）将有效降低云系统的计算成本；另一方面，与自包含解决方案相比，在云端（包括远程和边缘云）上执行其余 DNN 计算也将改善推理延迟，从而改善用户体验。


+ [DDPQN: An Efficient DNN Offloading Strategy in Local-Edge-Cloud Collaborative Environments](https://ieeexplore.ieee.org/abstract/document/9555248)
  + 2021 IEEE TSC
  + 我们考虑DNN模型的划分和卸载，设计了一种新的优化方法，用于资源受限的本地-边缘-云协同环境中大规模DNN模型的并行卸载。
  + 结合耦合协调度和节点平衡度，提出一种改进的双决斗优先深度Q网络（DDPQN）算法，获得DNN卸载策略。


+ [Reinforcement Learning Based Collaborative Inference and Task Offloading Optimization for Cloud-Edge-End Systems](https://ieeexplore.ieee.org/document/10651115)
  + 2023 IJCNN
  + 旨在通过联合优化动态环境下的任务卸载、模型划分和资源分配来减少 DNN 任务的长期平均端到端延迟。
  + 提出了一种新的可选分区点压缩算法，该算法根据层的输出特征选择高质量的分区点以降低模型划分的难度。为了提高代理的收敛性能，我们提出了一种基于强化学习的协同推理优化算法。
