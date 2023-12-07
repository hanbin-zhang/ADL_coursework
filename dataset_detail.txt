Files provided here are:
 - dataset.py    - Code for loading the audio samples and annotations from the annotation files of MagnaTagATune
 - evaluation.py - Code to evaluate your model output on the validation set. The model outputs should be a list of tensors of length 4332 (size of val). All tensors in the list should be of size 50 (number of classes).
 - MagnaTagATune.zip - The numpy arrays for all the audio samples, as well as the .pkl files for the annotations

**NOTE:** It will be much quicker to directly copy the data on BC4 with `cp -r /mnt/storage/scratch/jc17360/MagnaTagATune /mnt/storage/scratch/<user>` where you replace `<user>` with your university username. The .zip file is mainly here as a backup. Also, please store
the data in you **scratch** space, and **not** your home space. Your scratch directory is where all data should be stored, and your code in your home directory should point to the scratch space.