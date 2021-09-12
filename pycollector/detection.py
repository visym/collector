import os
import sys
import vipy
import numpy as np
import heyvi.detection
assert heyvi.version.is_at_least('0.0.3')


class TorchNet(heyvi.detection.TorchNet):
    pass

class FaceDetector(heyvi.detection.FaceDetector):
    pass
    
class Yolov5(heyvi.detection.Yolov5):
    pass
    
class Yolov3(heyvi.detection.Yolov3):
    pass
    
class ObjectDetector(heyvi.detection.ObjectDetector):
    pass

class MultiscaleObjectDetector(heyvi.detection.MultiscaleObjectDetector):  
    pass
    
class VideoDetector(heyvi.detection.VideoDetector):  
    pass
                       
class MultiscaleVideoDetector(heyvi.detection.MultiscaleVideoDetector):
    pass

class VideoTracker(heyvi.detection.VideoTracker):
    pass

class FaceTracker(heyvi.detection.FaceTracker):
    pass
    
class MultiscaleVideoTracker(heyvi.detection.MultiscaleVideoTracker):
    pass

class Proposal(heyvi.detection.Proposal):
    pass
    
class VideoProposal(heyvi.detection.VideoProposal):
    pass

class FaceProposalRefinement(heyvi.detection.FaceProposalRefinement):
    pass

class TrackProposalRefinement(heyvi.detection.TrackProposalRefinement):
    pass
    
class VideoProposalRefinement(heyvi.detection.VideoProposalRefinement):
    pass

class ActorAssociation(heyvi.detection.ActorAssociation):
    pass

