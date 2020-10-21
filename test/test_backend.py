import pycollector.video
import pytest 

def _test_video():
    raise ValueError('FAILED')
    v = pycollector.video.Video('A3A05DEF-4E8B-4650-B6D2-71BF43AD18D8')
    assert v.videoid() == 'A3A05DEF-4E8B-4650-B6D2-71BF43AD18D8'
    print('[test_video]: PASSED')

    
if __name__ == "__main__":
    _test_video()
