# VideoFaceDetector

1. KeyFrameGetter.py: Extract key frames from input movies.

2. FaceDetector.py: Detect the face in each key frame which extracts from the movie

## What you Need...

You need:

- Numpy
- OpenCV

Demos are showed in Main-Function.



## How to Begin...

### KeyFrameGetter.py

1. Put your video in `./video/`,if you don't have the dir, please create yourself.

2. (Option)You can set params in:

   1. ```
      kfg = KeyFrameGetter(source_path, dir_path, 100)  # 100 means extract 1 key-frame in each 100 frames
      ```

   2. ```
      kfg.load_diff_between_frm(alpha=0.07) # alpha means the difference parameter
      ```

3. Then you can click `run` and get your result in `./img/`



## My Thoughts...
https://segmentfault.com/a/1190000022192846

