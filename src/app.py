import os
import time
import requests
from camera_processor.camera_processor import CameraProcessor


def trigger_sprinkler(last_tripped, timeout=5.0):
    """This function is executed when an object is detected"""
    if time.time() > (last_tripped + timeout):
        requests.get(url + '/off/')
    url = 'http://10.0.0.79:8000'
    detection.identify()
    if not detection.is_person():
        requests.get(url + '/off/')
    else:
        url = 'http://10.0.0.79:8000'
        requests.get(url + '/on/')


def get_sprinkler_url():
    """Returns the URL of the sprinkler"""
    return 'http://10.0.0.79:8000'


def turn_sprinkler_on():
    """Turns the sprinkler on"""
    print('turning sprinkler on')
    requests.get(get_sprinkler_url() + '/on/')


def turn_sprinkler_off():
    """Turns the sprinkler off"""
    print('turning sprinkler off')
    requests.get(get_sprinkler_url() + '/off/')

def get_stream_uri():
    """Gets the stream URI from an environment variable"""
    try:
        stream_uri = os.environ.get('STREAM_URI')
        print('Stream URI: {}'.format(stream_uri))
        return stream_uri
    except KeyError:
        print('STREAM_URI not defined')
        exit(1)


def main():
    """Run the container"""
    stream_uri = get_stream_uri()
    processor = CameraProcessor(stream_uri, 'default camera')
    sprinkler_is_on = False
    no_detections_timeout = 0
    person_timeout = 5 # When a person is spotted, stay on for this amount of seconds
    last_person_seen = int(time.time())
    try:
        while True:
            raw_detections = processor.get_detections()
            for d in raw_detections:
                d.identify()
            detections = [x for x in raw_detections if x.is_person()]
            if not detections:
                if int(time.time()) < (last_person_seen + person_timeout):
                    continue
                no_detections_timeout = no_detections_timeout + 1
                if no_detections_timeout > 30:
                    if sprinkler_is_on:
                        turn_sprinkler_off()
                        sprinkler_is_on = False
                    no_detections_timeout = 0
            for detection in detections:
                if not sprinkler_is_on:
                    sprinkler_is_on = True
                    last_person_seen = int(time.time())
                    turn_sprinkler_on()   
    except KeyboardInterrupt:
        print('Exiting')


if __name__ == '__main__':
    main()
