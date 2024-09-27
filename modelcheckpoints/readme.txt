This directory holds the publicly released models of both our work (SynSin) and some of the baselines.

It is organised into directories containing models trained on the given dataset:
- kitti [1]
- realestate [2]
- mp3d (matterport) [3]

[1] http://www.cvlibs.net/datasets/kitti/
[2] https://google.github.io/realestate10k/
[3] https://niessner.github.io/Matterport/

The models released are:
1. zbufferpts.pth (SynSin [3]), synsin.pth (SynSin [3]). zbufferpts.pth is trained for longer than synsin.pth (250K vs 50K iterations)
2. zbufferpts_invdepth.pth (SynSin [3]), synsin_invdepth.pth (SynSin [3]). SynSin but where the depth is predicted as 1 / (Sigmoid(X) * C1 + C2))
5. viewappearance.pth (a model based on [1])
6. tatarchenko.pth (a model based on [2])

[1] Zhou, Tinghui, et al. "View synthesis by appearance flow." ECCV, 2016.

[2] Dosovitskiy, Alexey, et al. "Learning to generate chairs with convolutional neural networks." CVPR, 2015.

[3] Wiles, Olivia, et al. "SynSin: End-to-end View Synthesis from a Single Image." CVPR, 2020
