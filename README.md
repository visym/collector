VISYM COLLECTOR
-------------------
[![PyPI version](https://badge.fury.io/py/pycollector.svg)](https://badge.fury.io/py/pycollector) [![pycollector unit tests](https://github.com/visym/collector/workflows/pycollector%20unit%20tests/badge.svg)](https://github.com/visym/collector/actions?query=workflow%3A%22pycollector+unit+tests%22)

Live Datasets for Visual AI    
URL: https://github.com/visym/collector/    
Datasets: https://visym.github.io/collector/    

[Visym Collector](https://visym.com/collector) is a global platform for collecting large scale consented video datasets of people for visual AI applications. Collector is able to record, annotate and verify custom video datasets of rarely occuring activities for training visual AI systems, at an order of magnitude lower cost than existing methods. Our distributed data collection team is spread over five continents and fifty countries to collect unbiased datasets for global visual AI applications.
   
Visym Collector provides:  

* On-demand collection of rare classes  
* Simultaneous video recording, annotation and verification into a single unified platform  
* Touchscreen UI for live annotation of bounding boxes, activity clips and object categories
* Consented videos of people for ethical dataset construction with in-app face anonymization
* Python tools for hard negative mining and live model testing in PyTorch


Requirements
-------------------
python 3.*    
[ffmpeg](https://ffmpeg.org/download.html) (required for videos)    
[vipy](https://github.com/visym/vipy), torch, boto3, pandas


Installation
-------------------

```python
pip install pycollector
```

Quickstart
-------------------

<a href="https://apps.apple.com/app/id1498077968"><img alt="iOS" src="https://developer.apple.com/app-store/marketing/guidelines/images/badge-download-on-the-app-store.svg" height="50"/></a>  <a href="https://play.google.com/store/apps/details?id=com.visym.collector"><img alt="Android" src="https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg" height="50"/></a>


* **Install.** Get the Visym Collector app and sign-up as a new user in the public beta.

* **Collect.**  Collect a labeled video using your own collections in the mobile app
* **Visualize**.  Download your annotated videos and visualize them using the python tools:

```python
import pycollector.video
v = pycollector.video.last().show()
```

* **Test.** Convert to a 64x3x224x224 PyTorch tensor for testing with your convolutional network:

```python
t = v.clip(0,64).activitytube(maxdim=224).torch()
```

* **Repeat.**  Collect more videos like those your network got wrong for training, or let our collection team help you!



The [demos](https://github.com/visym/collector/tree/master/demo) will provide additional useful tutorials to help you get started.


