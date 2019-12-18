import util
import os
import shutil

# cityscape icin eklendi.
with open("../datasets/cityscapes-cycle/f_names.txt") as f:
    f_names = f.read().split('\n')

result_dir = "/home/retina/Research/samhi/CycleGAN/results/cityscapes_label2photo_pretrained_docker/latest_test/images/"
img_names = os.listdir(result_dir)

for img_name in img_names:

    index = int(os.path.basename(os.path.splitext(img_name)[0]).split("_")[0]) - 1  # cityscapes

    image_name = os.path.splitext(f_names[index])[0] + ".png"  # cityscape icin eklendi
    shutil.move(os.path.join(result_dir, img_name), os.path.join(result_dir, image_name))
