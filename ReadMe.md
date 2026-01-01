# Live MoCap

![Ganyu-ji](images/ganyu_ji.gif)

## Requirements

* python>=3.12
    * mediapipe
    * pytorch (cpu version is ok)
* blender >= 3.0 (for reading assets and binding animation)

## 安装依赖

推荐使用 uv。

```bash
$ uv venv
$ python .venv/bin/activate_this.py
$ uv sync
```

## 模型准备

到 https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index?hl=zh-cn#models 下载 full 模型到项目目录下。

## 使用示例

```bash
$ python mocap.py --blend assets/mixamo.blend --video test.mp4
```

运行后会出现一个窗口，使用 R 键旋转，矫正视频旋转，按空格键确定。

## How to use

1.  Prepare your character model
    
    Currently this script uses **Blender** to load model skeleton and bind animation. Your model should be saved as .blend file. 
    
    You may edit your model to assure that

    * Model must be in rest pose (clear all bone rotation/translation/scale in pose mode). And the rest pose should be close to T pose. 
    
    * Clear previous bone animation data and constraints.
    
    * Name related bones as below (in lower case). You may refer the mixamo example [assets/mixamo.blend](assets/mixamo.blend) to see to name the bones so that they can be recogonzed and binded.
    ![](images/mixamo.png)

    * Save the model as `.blend` file somewhere.

2.  Run the script `mocap.py`.

    ```
    python mocap.py --blend your_character_model.blend --video your_video.mp4 [other options] 
    ```

    The program will read and capture motion from the video, save the animation data, and then open Blender and bind the animation to your charactor model. After everything is done, you should be able to see the Blender window with your character already animated.

# Future work

* Now working on face capture.
