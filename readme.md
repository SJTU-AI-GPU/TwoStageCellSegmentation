# TwoStageCellSegmentation

pytorch model files

- Resnet_base_50_model.py
- Resnet_base_50_parts.py

training image of sample [3 class] [the picture is watermarked temporarily]

- ./train_image_sample_3c/*

Original sample [4 class] color is stated as following

    Unlabelled  =   np.array([0,     0,      0   ]) #black, background
    A           =   np.array([80,    0,      255 ]) #red, inflammatory cells 
    B           =   np.array([192,   255,    128 ]) #Light blue, nuclei
    C           =   np.array([64,    255,    64  ]) #green, cytoplasm      
training model file[for the first stage of our two stage model]

- https://drive.google.com/file/d/1KjoOBvX4M__kkwiVvUQaZT86DnBd1ZYE/view?usp=sharing