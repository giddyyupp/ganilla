import os
import torch
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html


torch.cuda.empty_cache()


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    # added for cityscapes.
    if opt.cityscapes:
        with open(opt.cityscape_fnames) as f:
            f_names = f.read().split('\n')


    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        mess = 'processing (%04d)-th of %04d image... %s' % (i, len(dataset), img_path[0])
        print(mess)
        # Open a file with access mode 'a'
        file_object = open('progress.txt', 'a')
        # Append 'hello' at the end of file
        file_object.write(mess+'\n')
        # Close the file
        file_object.close()
        save_images(webpage, visuals, img_path,  save_both=opt.save_both, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        
        if(opt.remove_images):
            os.remove(img_path[0])
            print('removed image', img_path[0])
        
    # save the website
        webpage.save()
