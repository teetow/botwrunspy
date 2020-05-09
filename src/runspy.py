import cv2
import numpy as np

from agents.buffspy import parse_buffs
from agents.mobspy import parse_mob
from agents.splitspy import parse_splits
from lib.framegrab import grabframe, grabframes
from lib.imgproc import crop_rect
from lib.profiles import PROFILES, Names
from lib.utils import dump

url_joe_talus = "https://www.twitch.tv/videos/613665739?t=00h56m05s"  # joed fighting a Talus
img_joe_talus = "testdata/joed_talus.png"
url_dj = "https://www.twitch.tv/videos/613398148?t=04h53m23s"  # dj with loading screen
img_dj = "testdata/dj_newlayout.png"


def test_joed_talus():
    p = PROFILES[Names.JoeDunff]
    f = cv2.imread(img_joe_talus)
    f = crop_rect(f, p.game_crop)
    print(parse_mob(f))


def test_average_frames():
    # experimental
    p = PROFILES[Names.DJ]
    fs = grabframes(url_dj)
    first_frame = fs[0]
    f = np.zeros((first_frame.shape[0], first_frame.shape[1], 3), 'float32')
    [cv2.accumulateSquare(x.astype('float32')/len(fs), f) for x in fs[1:]]
    f = (f/f.max())*256
    f = f.astype('uint8')


def test_dj_splits():
    p = PROFILES[Names.DJ]
    f = grabframe(url_dj)
    splits = parse_splits(f, p)
    [print(str(x)) for x in splits]


def test_things():
    p = PROFILES[Names.Specs]
    
    frame = grabframe(p.url)
    game_frame = crop_rect(frame, p.game_crop)
    
    splits = parse_splits(frame, p)
    buffs = parse_buffs(game_frame, p)
    mob = parse_mob(game_frame)
    
    print([str(x) for x in splits])
    print(str(buffs))
    print(str(mob))


if __name__ == "__main__":
    test_things()
