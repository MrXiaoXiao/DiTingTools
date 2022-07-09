# DiTingTools [谛听--用于天然地震数据处理的深度学习工具包]

## Description [简介]

DiTingTools aims to provide modules for developing, training, evaluating, and applying deep-learning-based methods to solve data processing tasks in seismology. By combining large datasets such as DiTing, STEAD, INSTANCE, and so on, we hope to build solid benchmarks for various seismological studies so that deep-learning-enabled data processing methods can be better evaluated and improved. And eventually, they could become more practical and powerful in seismological research. In that sense, DiTingTools is similar to an excellent previous work named 'Seisbench' (Woollam et al., 2022; https://github.com/seisbench/seisbench). Yet, in technique aspects, DiTingTools uses TensorFlow while Seisbench uses PyTorch. Besides, their code structures are quite different as well. We would like DiTingTools to be a beneficial supplement to the existing works and contribute to the development of deep-learning-enabled seismology.

谛听工具包主要为开发、训练、测评和应用基于深度学习的天然地震数据处理方法提供帮助。我们希望通过综合多个大型地震数据集，为多个地震学数据处理任务提供基准测试，以更好改进这些基于深度学习的处理方法，使它们更加实用和有效，更好的为地震学研究服务。在这一点上，DiTingTools与此前优秀的工具包Seisbench (Woollam 等人., 2022; https://github.com/seisbench/seisbench )相似。DiTingTools基于TensorFlow而SeisBench基于PyTorch,此外二者的代码结构设计也完全不同。我们希望谛听工具包可以作为已有工作的有益补充，并更好帮助人工智能算法在地震学研究中落地应用。

## Getting Started [教程]

Right now, there are several example notebooks: 

目前有如下例子：

(1) How to read DiTing dataset, as well as STEAD, INSTANCE, NEIC, and SCSN-JGR datasets;

(1) 如何读取谛听数据集, 以及STEAD、INSTANCE、NEIC、SCSN-JGR数据集；

(2) How to train an earthquake detection and phase picking model with DiTingTools.

(2) 如何使用DiTingTools训练地震检测和震相拾取模型；

(3) How to train a first-motion polarity determination model with DiTingTools. 

(3) 如何使用DiTingTools训练一个初动极性判别模型。

(4) How to train an azimuth regression model with DiTingTools. [To be updated before 12th Jul.]

(5) How to train a magnitude estimation model with DiTingTools. [To be updated before 12th Jul.]

More examples will be updated in the future.

### Dependencies [运行环境]

* tensorflow == 2.2.0
* keras == 2.4.3
* obspy
* scipy
* numpy
* pyyaml
* h5py
* tqdm
* scikit-learn

### Installing

For the demo version, just clone the project, unzip the code, and append it to your path. Please refer to the notebooks in the 'examples' folder.

对于目前原型版本，下载代码，解压缩并添加至程序路径即可。添加路径方式可以参照'examples'文件夹下的notebook。

## Notice [注意事项]

This module is undergoing rapid development. The demo version is for presentations at the DiTingCamp and USTC-seismo-training.

该模块仍在开发阶段。目前原型版本为谛听训练营和科大“地震学算法与程序培训班”使用。

The cleaning and update on the demo version will be finished before 15th Jul. 2022.

完整原型版代码整理将在7月15号之前上传完成。

The continuous development of DiTingTools is going to be a challenging long-term task. We welcome any suggestions or feedback.

谛听工具包的不断开发和完善是一个长期的事情，欢迎各位专家的建议和反馈。

## Version History

-- To be updated

## License

GPLv3.0

## Citation

We will upload a pre-print around Augest 2022, please check back then.

我们大概会在8月份上传一个pre-print以供引用，请届时再回来看下。[最近太忙了X_X]
