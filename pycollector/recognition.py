import heyvi
assert heyvi.version.is_at_least('0.0.3')


class ActivityRecognition(heyvi.recognition.ActivityRecognition):
    pass
    
class PIP_250k(heyvi.recognition.PIP_250k):
    pass

class PIP_370k(heyvi.recognition.PIP_370k):
    pass

class ActivityTracker(heyvi.recognition.ActivityTracker): 
    pass

