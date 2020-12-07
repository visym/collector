<a href="https://github.com/visym/collector" class="github-corner" aria-label="View source on GitHub"><svg width="80" height="80" viewBox="0 0 250 250" style="fill:#151513; color:#fff; position: absolute; top: 0; border: 0; right: 0;" aria-hidden="true"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path></svg></a>

[![Collector visualization](https://i3.ytimg.com/vi/HjNa7_T-Xkc/maxresdefault.jpg)](https://youtu.be/HjNa7_T-Xkc)

<a href="https://visym.com/collector"><img alt="iOS" src="https://developer.apple.com/app-store/marketing/guidelines/images/badge-download-on-the-app-store.svg" height="50"/></a>  <a href="https://visym.com/collector"><img alt="Android" src="https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg" height="50"/></a> 

[Contact us](mailto:info@visym.com?subject=[Visym Collector]: Beta Access) for access to the private beta of the mobile app!


## VISYM COLLECTOR

[Visym Collector](https://visym.com/collector) is a mobile app for collecting large scale, on-demand and consented video datasets of people for visual AI applications. Collector is able to record, annotate and verify custom video datasets of rarely occuring activities for training visual AI systems, at an order of magnitude lower cost than existing methods. Our distributed data collection team is spread over five continents and fifty countries to collect unbiased datasets for global visual AI applications.
   
Visym Collector provides:  

* On-demand collection of rare classes  
* Simultaneous video recording, annotation and verification into a single platform
* Touchscreen UI for live annotation of bounding boxes, activity clips and object categories
* Consented videos of people for ethical dataset construction with in-app face anonymization
* [Python tools](https://github.com/visym/collector) for hard negative mining, [dataset transformation](https://github.com/visym/vipy), active learning and live model testing in PyTorch

Our goal is to make all datasets freely available to the computer vision research community.


## Dataset Releases

<!-- Backup links:  https://htmlpreview.github.io/?https://github.com/visym/collector/blob/master/docs/pip_175k/trainset_small.html -->

* **People in Public - 175k.**  This dataset contains 184,379 video clips of 66 classes of activities performed by people in public places.  The activity labels are subsets of the 37 activities in the [Multiview Extended Video with Activities (MEVA)](https://mevadata.org) dataset and is consistent with the [Activities in Extended Video (ActEV)](https://actev.nist.gov/) challenge.   &nbsp;&nbsp;[[README]](pip_175k/README.md)&nbsp;&nbsp;
    * Visualization of [Training set random sample (87MB)](https://rawcdn.githack.com/visym/collector/5b051c625ef458417a16ed48d5a0693ef59fd9ff/docs/pip_175k/trainset_small.html),&nbsp;[full validation set (1.1GB)](https://dl.dropboxusercontent.com/s/8fp77nvxeywrq7f/pip_175k_valset.html)
    * People in Public - 175k - stabilized.  Background stabilized videos for pip-175k.  [[README]](pip_175k_stabilized/README.md)&nbsp;&nbsp;
* **People in Public - 250k - stabilized.**  This dataset contains 314,649 background stabilized videos clips of 66 classes of activities performed by people in public places.  &nbsp;&nbsp;[[README]](pip_250k_stabilized/README.md)&nbsp;&nbsp;

