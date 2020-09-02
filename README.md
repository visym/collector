VISYM COLLECTOR
-------------------
[![PyPI version](https://badge.fury.io/py/pycollector.svg)](https://badge.fury.io/py/pycollector)

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
python >= 3.3  
ffmpeg (required for videos)  
vipy, boto3, pandas, torch


Installation
-------------------

```python
pip install pycollector
```

Quickstart
-------------------
```python
import pycollector
```

The [demos](https://github.com/visym/collector/demo) provide useful notebook tutorials to help you get started.


