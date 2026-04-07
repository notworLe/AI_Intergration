# Face recognition

## Theory

# Usage step

**Note:** Make sure you have installed notworLeUtils

1. Preprare your dataset

- At root (face_recognition)
  `cd dataset`
- Add your dataset with structure as:

```
dataset
|
├── your_name/ (Identify your)
|───── raw/ (Your face images (only your face))
|
```

2. Train

- Convert your face images into embedding. It also uploads to my database in Neon  
  `python train.py`

3. Main

- Run the program
  `python main.py`
