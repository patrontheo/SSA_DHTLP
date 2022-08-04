# Inference

## Demo

In order to test the algorithm on some images, the `demo.py` can be used as follows:  
`python demo.py -d 0 config/wireframe.yaml weights/checkpoint.pth.tar path_to_img1.png path_to_img2.png`  
The arguments are detailed in [DHTLP](https://github.com/yanconglin/Deep-Hough-Transform-Line-Priors).  Replace -d 0 by -d "" if you have no GPU.

This will create new png files located in the same folder as the input images and containing the detected streaks. The displayed streaks are the one for which the confidence value is above a fixed threshold (0.94 here).

## Dataset inference

In order to perform the inference on a folder containing images, the file `process_folder.py` can be used as follows:  
`python process_folder.py -d 0 config/wireframe.yaml weights/checkpoint.pth.tar path_to_folder --plot`

The argument `--plot` is optional and can be removed to avoid creating images showing the detected lines.  

*path_to_folder* should be the path to a folder containing all the images (the images can be contained in different subfolders). e.g. `data/img` if the images are contained in the folder `img`.

This will create a *csv* file in *path_to_folder* containing the detected lines for all the images.

# Training

## Dataset processing  

In order to train the network, the images containing a streak should be annotated. The coordinates of the two endpoints of each streak are required. The images can be annotated quite easily using https://www.makesense.ai/.  
Two processing steps are needed to reach a dataset suitable for training. The first is here to make the dataset ready to be used by the processing step implemented by the network authors.

### Pre-processing  

*Warning* in the current implementation, the annotation has been done by putting 2 points at each endpoint of the streak. This assumes that there is only one streak in the image. To be more robust (i.e. to handle 2 streaks or more), the annotation should be done using the *line* tool of the mentionned website. Implementing this possibility would require to change a little bit the way how the annotations are parsed in the `construct_dataset.py` file.

In order to have a usable dataset, the images need to be processed through `construct_dataset.py`.  
This script take as input a folder containing:  
* A file named `labels.csv` and containing the annotated streak on all the images.
* A folder named `train_split` and containing the images of the training set (`fits` files)
* A folder named `val_split` and containing the images of the validation set (`fits` files)

The above script creates a new folder containing the `json` files needed to process the dataset.  

For now, the input image directory and the output directory are hardcoded in the `construct_dataset.py` file.  

### Processing  

The second processing step is straightforward to perform.  
The `dataset/wireframe.py` file should be called with the above *output* directory as the new *input* directory, and with a new *output* directory that will contain the completely processed dataset.  

`python dataset/wireframe.py preprocessed_dataset_folder ouptut_folder` 
The directory created by this step can be used for training.


## Training

In order to reach good results, it is strongly advised to start the training with the weights pretrained on the ShanghaiTech dataset. See [DHTLP](https://github.com/yanconglin/Deep-Hough-Transform-Line-Priors) to download the weights.  

The following command launch the training of the network:  
`python train.py -d 0 --identifier baseline config/wireframe.yaml`  

The file `config/wireframe.yaml` needs to be edited accordingly:  
* The batch size depends on the GPU memory size.
* `datadir` is the completely processed dataset.
* `resume_from` is the folder containing the pretrained weights (that should be named `checkpoint_latest.pth.tar`). This will also be the output folder if used.
* The learning rate is set here to 1e-6 but can be adjusted.
* The `max_epochs` is currently set to a very large number and the training is manually stopped when the validation loss keep increasing.


# sAP evaluation

In order to compute the sAP obtained on a given batch of images, the file `process_folder_sAP.py` can be used.  
To use it, the dataset needs to be pre-processed with the pre-processing step described above: a folder should contain a file named *valid.json* and a subfolder *images* containing all the images (This correspond to the folder 2 in the given sample dataset).

The following command can be used:  
`python process_folder_sAP.py -d 0 config/wireframe.yaml weights/checkpoint_best.pth.tar path_to_data_folder`  

This will create a file named `result_sAP.txt` that contains the results.




