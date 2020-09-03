VISYM COLLECTOR
-------------------
[![PyPI version](https://badge.fury.io/py/pycollector.svg)](https://badge.fury.io/py/pycollector) [![pycollector unit tests](https://github.com/visym/collector/workflows/pycollector%20unit%20tests/badge.svg)](https://github.com/visym/collector/actions?query=workflow%3A%22pycollector+unit+tests%22)

Live Datasets for Visual AI    
URL: https://github.com/visym/collector/    

Visym Collector is a global platform for collecting large scale consented video datasets of people for visual AI applications. Collector is able to record, annotate and verify custom video datasets of rarely occuring activities for training visual AI systems (e.g. activity detection), at an order of magnitude lower cost than existing methods. Our distributed data collection team is spread over five continents and fifty countries to collect unbiased datasets for global visual AI applications.
   
Visym Collector provides:  

* On-demand collection of rare classes  
* Simultaneous video recording, annotation and verification into a single unified platform  
* Consented videos of people for ethical dataset construction
* Python tools for hard negative mining and live model testing in PyTorch

[![Visym Collector Montage](http://i3.ytimg.com/vi/HjNa7_T-Xkc/maxresdefault.jpg)](https://youtu.be/HjNa7_T-Xkc)


Requirements
-------------------
python 3.*
ffmpeg (required for videos)  
vipy, boto3, pandas, torch


Installation
-------------------

```python
pip install pycollector
```

Quickstart
-------------------

<a href="https://visym.com/collector"><img alt="iOS" src="https://developer.apple.com/app-store/marketing/guidelines/images/badge-download-on-the-app-store.svg" height="50"/></a>  <a href="https://visym.com/collector"><img alt="Android" src="https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg" height="50"/></a>


**1. Install.** Get the Visym Collector app (contact us to join the private beta!)

**2. Collect.**  Collect a labeled video using the mobile app, then retrieve and visualize it using the python tools:

```python
import pycollector
v = pycollector.video.last()
v.show()
```

**3. Test.** Convert the labeled video to a 64x3x224x224 PyTorch tensor and test with your network

```python
net(v.maxsquare().crop().mindim(224).clip(0,64).torch())
```

**4. Repeat.**  Collect more videos like those your network got wrong, or let our collection team collect for you!



The [demos](https://github.com/visym/collector/tree/master/demo) will provide useful notebook tutorials to help you get started.


