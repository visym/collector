<iframe width="1280" height="720" position="absolute" top="0" left="0" src="https://www.youtube.com/embed/HjNa7_T-Xkc" title="Visym Collector: Consented and On-demand Visual Datasets" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## VISYM COLLECTOR

[Visym Collector](https://visym.com/collector) is a mobile app for collecting large scale, on-demand and consented video datasets of people for visual AI applications. Collector is able to record, annotate and verify custom video datasets of rarely occuring activities for training visual AI systems, at an order of magnitude lower cost than existing methods. Our distributed data collection team is spread over five continents and fifty countries to collect unbiased datasets for global visual AI applications.
   
Visym Collector provides:  

* On-demand collection of rare classes  
* Simultaneous video recording, annotation and verification into a single platform
* Touchscreen UI for live annotation of bounding boxes, activity clips and object categories
* Ethical and Consented videos of people for dataset construction with face and license plate anonymization
<!-- * [Python tools](https://github.com/visym/collector) for hard negative mining, [dataset transformation](https://github.com/visym/vipy), active learning and live model testing in PyTorch -->

Our goal is to make all datasets freely available to the computer vision research community.


## Dataset Releases

<!-- Backup links:  https://htmlpreview.github.io/?https://github.com/visym/collector/blob/master/docs/pip_175k/trainset_small.html -->

* **[VisymScenes](https://doppelgangers25.github.io/doppelgangers_plusplus/).**  This dataset includes residential areas, landmarks, historical sites, business districts, and more. This dataset contains 258K images with GPS/IMU metadata, recorded at 149 sites in 42 cities and 15 countries. 
* **[Consented Activities of People](https://visym.github.io/cap).**  This dataset contains 1.45M video clips of 512 classes of activities of daily life.  Videos are recorded from mobile devices around the world from a third person viewpoint looking down on the scene from above, containing consented subjects performing every day activities.    
* **[People in Public - 370k - stabilized](pip_370k_stabilized/README.md).**  This dataset contains 405,781 background stabilized video clips of 63 classes of activities performed by people in public places.  
* **[People in Public - 250k - stabilized](pip_250k_stabilized/README.md).**  This dataset contains 314,332 background stabilized video clips of 66 classes of activities performed by people in public places.
* **[People in Public - 175k](pip_175k/README.md).**  This dataset contains 184,379 video clips of 66 classes of activities performed by people in public places.  The activity labels are subsets of the 37 activities in the [Multiview Extended Video with Activities (MEVA)](https://mevadata.org) dataset and is consistent with the [Activities in Extended Video (ActEV)](https://actev.nist.gov/) challenge.  
    * Visualization of [Training set random sample (87MB)](https://rawcdn.githack.com/visym/collector/5b051c625ef458417a16ed48d5a0693ef59fd9ff/docs/pip_175k/trainset_small.html),&nbsp;[full validation set (1.1GB)](https://dl.dropboxusercontent.com/s/8fp77nvxeywrq7f/pip_175k_valset.html)
    * [People in Public - 175k - stabilized](pip_175k_stabilized/README.md).  Background stabilized videos for pip-175k.  


