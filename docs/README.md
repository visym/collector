<div class="container" position="relative" width="100%" height="0" padding-bottom="56.25%"><iframe width="100%" height="100%" position="absolute" top="0" left="0" src="https://www.youtube.com/embed/HjNa7_T-Xkc" title="Visym Collector: Consented and On-demand Visual Datasets" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>

<a href="https://apps.apple.com/app/id1498077968"><img alt="iOS" src="https://developer.apple.com/app-store/marketing/guidelines/images/badge-download-on-the-app-store.svg" height="50"/></a>  <a href="https://play.google.com/store/apps/details?id=com.visym.collector"><img alt="Android" src="https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg" height="50"/></a> <a href="https://github.com/visym/collector"><img alt="Github" src="https://github.com/fluidicon.png" height="50"/></a> <a href="https://visym.com/collector"><img alt="Visym Labs" src="https://www.visym.com/labs/images/visym_logo_black_notext.png" height="50"/></a> 

## VISYM COLLECTOR

[Visym Collector](https://visym.com/collector) is a mobile app for collecting large scale, on-demand and consented video datasets of people for visual AI applications. Collector is able to record, annotate and verify custom video datasets of rarely occuring activities for training visual AI systems, at an order of magnitude lower cost than existing methods. Our distributed data collection team is spread over five continents and fifty countries to collect unbiased datasets for global visual AI applications.
   
Visym Collector provides:  

* On-demand collection of rare classes  
* Simultaneous video recording, annotation and verification into a single platform
* Touchscreen UI for live annotation of bounding boxes, activity clips and object categories
* Ethical and Consented videos of people for dataset construction with in-app face anonymization
* [Python tools](https://github.com/visym/collector) for hard negative mining, [dataset transformation](https://github.com/visym/vipy), active learning and live model testing in PyTorch

Our goal is to make all datasets freely available to the computer vision research community.


## Dataset Releases

<!-- Backup links:  https://htmlpreview.github.io/?https://github.com/visym/collector/blob/master/docs/pip_175k/trainset_small.html -->

* **[People in Public - 370k - stabilized](pip_370k_stabilized/README.md).**  This dataset contains 405,781 background stabilized video clips of 63 classes of activities performed by people in public places.  
* **[People in Public - 250k - stabilized](pip_250k_stabilized/README.md).**  This dataset contains 314,332 background stabilized video clips of 66 classes of activities performed by people in public places.
* **[People in Public - 175k](pip_175k/README.md).**  This dataset contains 184,379 video clips of 66 classes of activities performed by people in public places.  The activity labels are subsets of the 37 activities in the [Multiview Extended Video with Activities (MEVA)](https://mevadata.org) dataset and is consistent with the [Activities in Extended Video (ActEV)](https://actev.nist.gov/) challenge.  
    * Visualization of [Training set random sample (87MB)](https://rawcdn.githack.com/visym/collector/5b051c625ef458417a16ed48d5a0693ef59fd9ff/docs/pip_175k/trainset_small.html),&nbsp;[full validation set (1.1GB)](https://dl.dropboxusercontent.com/s/8fp77nvxeywrq7f/pip_175k_valset.html)
    * [People in Public - 175k - stabilized](pip_175k_stabilized/README.md).  Background stabilized videos for pip-175k.  


