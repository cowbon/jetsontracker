import unittest
from CameraManager import .

test_str = '
{
    "sdp":[
	{
	    "link": "192.168.216.138:5540/ch0"
	}
	{
	    "link": "150.113.195.211:8080/camera.sdp"
	}
    ]
    "video":[
	{
	    "
	}
    ]
}'


class CameraManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.cm = CameraManager('config.json')

    def test_device_count(self):
        self.assertEquals(self.cm.device_count(), 4)
