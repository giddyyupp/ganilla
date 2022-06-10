import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import wandb
import torch

#WB_PROJECT = "ganilla"
WB_PROJECT = "ganimals"
WB_USER = "stacey"

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
    wandb.init(project=WB_PROJECT, entity=WB_USER, name=opt.name)
    # log model
    model_at = wandb.Artifact("ganilla_"+opt.name, type="model")
    for name in model.model_names:
        if isinstance(name, str):
            save_path = 'net_%s.pth' % (name)
            net = getattr(model, 'net' + name)
            if len(model.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), save_path)
                net.cuda(model.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)
    model_at.add_file(save_path)
    wandb.run.log_artifact(model_at)
 
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    # cityscape icin eklendi.
    if opt.cityscapes:
        with open(opt.cityscape_fnames) as f:
            f_names = f.read().split('\n')
    img_data = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 50 == 0:   
            source_image = wandb.Image(img_path[0])
            print(visuals["fake_B"].shape)
            dest_image = wandb.Image(visuals["fake_B"].squeeze(0).permute(1, 2, 0).cpu().numpy())
            img_data.append([source_image, dest_image])
        if opt.cityscapes:
            index = int(os.path.basename(img_path[0]).split("_")[0]) - 1  # cityscapes
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        if not opt.cityscapes:
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, citysc=False)
        else:
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                        f_name=f_names[index], citysc=True)  # cityscapes
    # save the website
    wandb.log({"samples" : wandb.Table(data=img_data, columns=["source", "result"])})
    webpage.save()
