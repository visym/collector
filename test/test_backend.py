import pycollector.backend
import pycollector.video


def test_video():
    v = pycollector.video.Video('A3A05DEF-4E8B-4650-B6D2-71BF43AD18D8')
    assert v.videoid() == 'A3A05DEF-4E8B-4650-B6D2-71BF43AD18D8'
    print('[test_video]: PASSED')

    
def test_backend():
    P = pycollector.backend.Prod()
    assert P.s3_bucket() == 'diva-prod-data-lake174516-visym'
    print('[test_backend]: PASSED')

    
if __name__ == "__main__":
    test_video()
    test_backend()
    
