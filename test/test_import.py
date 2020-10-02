import os

def test_import():
    import pycollector
    import pycollector.backend    
    import pycollector.dataset    
    import pycollector.detection
    import pycollector.globals
    import pycollector.label    
    import pycollector.project    
    import pycollector.util
    import pycollector.user    
    import pycollector.version
    import pycollector.video
    print('[test_import]: PASSED')


if __name__ == "__main__":
    test_import()
    
