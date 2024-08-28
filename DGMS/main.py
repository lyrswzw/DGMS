import torch.optim as optim
from model_dgms import *
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label


print(torch.version.cuda)
torch.cuda.set_per_process_memory_fraction(fraction=0.8, device=None)
import time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    dataset = 'pascnn'  # pascnn  wikipedia xmedia
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    phase = "train"

    DATA_DIR = './data/' + dataset + '/'
    MAX_EPOCH = 100  # 220
    MAX_EPOCHGAN = 100  # 200
    batch_size = 100  # 100
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0.01

    print('...Data loading is beginning...')
    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)
    print('...Data loading is completed...')

    model_ft = CrossGAT(batch_size, img_input_dim=input_data_par['img_dim'],
                        text_input_dim=input_data_par['text_dim'],
                        output_dim=input_data_par['num_class']).to(device)
    num_params = count_parameters(model_ft)
    print(f"模型参数数量：{num_params}")
    params_to_update = list(model_ft.parameters())

    dis_ft = DiscriminatorV().to(device)
    params_dis = list(dis_ft.parameters())

    gen_ft = GeneratorV().to(device)
    params_gen = list(gen_ft.parameters())

    disT_ft = DiscriminatorT().to(device)
    params_disT = list(disT_ft.parameters())

    genT_ft = GeneratorT().to(device)
    params_genT = list(genT_ft.parameters())
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
    optimizer_dis = optim.Adam(params_dis, lr=lr, betas=betas)
    optimizer_gen = optim.Adam(params_gen, lr=lr, betas=betas, weight_decay=weight_decay)
    optimizer_disT = optim.Adam(params_disT, lr=lr, betas=betas)
    optimizer_genT = optim.Adam(params_genT, lr=lr, betas=betas, weight_decay=weight_decay)

    if phase == "test":
        model_ft.load_state_dict(torch.load('model/' + dataset + '_pre-model_gat_loss.pt'))
        gen_ft.load_state_dict(torch.load('model/' + dataset + '_pre-model_gen.pt'))
        genT_ft.load_state_dict(torch.load('model/' + dataset + '_pre-model_genT.pt'))
    elif phase == "train":
        print('...Training is beginning...')
        # Train and evaluate
        start = time.time()
        genT_ft, gen_ft, model_ft = train_model(genT_ft, disT_ft, gen_ft, dis_ft, model_ft, data_loader,
                                                optimizer_genT, optimizer_disT, optimizer_gen, optimizer_dis,batch_size,
                                                optimizer, dataset, num_epochs=MAX_EPOCH, num_epochsGAN=MAX_EPOCHGAN)
        end = time.time()
        print('Total train time:{:.4f} min'.format((end - start) / 60))
        print('...Training is completed...')
    else:
        print("Please select the phase train or test")

    print('...Evaluation on testing data...')
    print('-------------------------------------')
    print(dataset)
    print('-------------------------------------')
    label = torch.argmax(torch.tensor(input_data_par['label_val']), dim=1)

    ################################
    _, _, _, _, _, _, _, _, imgout, textout, _, _, _, _, _= model_ft(
        torch.tensor(input_data_par['img_val']).to(device),
        torch.tensor(input_data_par['text_val']).to(device))

    imgout = imgout.detach().cpu()
    textout = textout.detach().cpu()

    img_to_txt_gat = fx_calc_map_label(imgout, textout, label)
    print('attion...Image to Text MAP = {}'.format(img_to_txt_gat))

    txt_to_img_gat = fx_calc_map_label(textout, imgout, label)
    print('attion...Text to Image MAP = {}'.format(txt_to_img_gat))

    print('attion...Average MAP = {}'.format(((img_to_txt_gat + txt_to_img_gat) / 2.)))


    #################################
    gen_img = gen_ft(torch.tensor(input_data_par['img_val']).to(device))
    gen_txt = genT_ft(torch.tensor(input_data_par['text_val']).to(device))
    gen_img = gen_img.detach().cpu()
    gen_txt = gen_txt.detach().cpu()

    img_to_txt = fx_calc_map_label(gen_img, gen_txt, label)
    print('DGMS...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(gen_txt, gen_img, label)
    print('DGMS...Text to Image MAP = {}'.format(txt_to_img))

    print('DGMS...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
