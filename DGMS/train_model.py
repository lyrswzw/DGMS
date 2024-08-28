from __future__ import print_function
from __future__ import division
import torch
import copy
from evaluate import fx_calc_map_label
import numpy as np

def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels):
    term1 = ((view1_predict - labels.float()) ** 2).sum(1).sqrt().mean()
    term2 = ((view2_predict - labels.float()) ** 2).sum(1).sqrt().mean()

    term3 = ((view2_feature - view1_feature) ** 2).sum(1).sqrt().mean()
    term4 = ((view2_predict - view1_predict) ** 2).sum(1).sqrt().mean()

    im_loss = (term1 + term2) * 5 + term3 + term4
    return im_loss

def HSIC(c_v, c_w):
    N = c_v.size(0)
    H = torch.ones((N, N), dtype=torch.float32, device=c_v.device) * (-1/N) + torch.eye(N, device=c_v.device)
    K_1 = torch.matmul(c_v, c_v.transpose(0, 1))
    K_2 = torch.matmul(c_w, c_w.transpose(0, 1))
    rst = torch.matmul(K_1, H)
    rst = torch.matmul(rst, K_2)
    rst = torch.matmul(rst, H)
    rst = torch.trace(rst)
    return rst

def loss_regularization(adjA, adjB):
    term1 = ((adjA ** 2.0).sum()).sqrt().mean()
    term2 = ((adjB ** 2.0).sum()).sqrt().mean()
    return term1 + term2

def calc_loss2(view1_feature, view2_feature, view1_predict, labels_1, tau=12):
    term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean()

    term2 = ((view1_feature - view2_feature) ** 2).sum(1).sqrt().mean()
    im_loss = term1 * 5 + term2 * tau
    return im_loss

def normalize_adj(adj, mask = None):
    if mask is None:
        mask = adj
    rowsum = np.sum(adj, axis=1)  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)  # D^-0.5
    return np.matmul(np.matmul(d_mat_inv_sqrt, mask), d_mat_inv_sqrt)


def generate_adj(labels):
    y_single = np.argmax(np.vstack((labels)), 1)
    y_single = y_single.reshape(y_single.shape[0], 1)
    mask_initial = np.matmul(y_single, np.ones([1, y_single.shape[0]], dtype=np.int32)) - \
                   np.matmul(np.ones([y_single.shape[0], 1], dtype=np.int32), np.transpose(y_single))
    adj = (np.equal(mask_initial, np.zeros_like(mask_initial)).astype(np.float32) - np.identity(
        mask_initial.shape[0]).astype(np.float32))+ np.identity(mask_initial.shape[0]).astype(
        np.float32)
    mask = (np.equal(mask_initial, np.zeros_like(mask_initial)).astype(np.float32) - np.identity(
        mask_initial.shape[0]).astype(np.float32)) + np.identity(mask_initial.shape[0]).astype(
        np.float32)
    mask = normalize_adj(adj, mask)
    adj = torch.from_numpy(mask)
    return adj

def train_gse(model, batch_size, data_loaders, optimizer, dataset, num_epochs=100):
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects_img = 0
            running_corrects_txt = 0
            index = 0

            for imgs, txts, labels in data_loaders[phase]:
                if imgs.shape[0] != batch_size:
                    cha = batch_size-imgs.shape[0]
                    imgs = torch.cat([imgs, imgs[imgs.shape[0] - cha:].clone()], dim=0)
                    txts = torch.cat([txts, txts[txts.shape[0] - cha:].clone()], dim=0)
                    labels = torch.cat([labels, labels[labels.shape[0] - cha:].clone()], dim=0)
                index = index + 1

                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")

                optimizer.zero_grad()  # 清空梯度

                with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()

                    optimizer.zero_grad()

                    imgs1, imgs2, texts1, texts2, imgc1, imgc2, textc1, textc2, img_all, text_all, img_predict, text_predict, adj_A, adj_B, adj_C= model(imgs, txts)
                    ###########################################################################################################
                    loss1 = calc_loss(img_all, text_all, img_predict, text_predict, labels)

                    loss3 = loss_regularization(adj_A, adj_B) * 0.1
                    loss4 = HSIC(adj_A, adj_B) * 0.1
                    loss = loss1 + loss3 + loss4

                    img_preds = img_predict
                    txt_preds = text_predict

                    if phase == 'train':
                        loss.requires_grad_(True)
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

                running_corrects_img += torch.sum(torch.argmax(img_preds, dim=1) == torch.argmax(labels, dim=1))

                running_corrects_txt += torch.sum(torch.argmax(txt_preds, dim=1) == torch.argmax(labels, dim=1))

            epoch_loss = running_loss / len(data_loaders[phase].dataset)

            t_imgsa, t_txtsa, t_imgs, t_txts, t_labels = [], [], [], [], []

            with torch.no_grad():
                for imgs, txts, labels in data_loaders['test']:
                    if imgs.shape[0] < batch_size:
                        cha = batch_size - imgs.shape[0]
                        imgs = torch.cat([imgs, imgs[imgs.shape[0] - cha:].clone()], dim=0)
                        txts = torch.cat([txts, txts[txts.shape[0] - cha:].clone()], dim=0)
                        labels = torch.cat([labels, labels[labels.shape[0] - cha:].clone()], dim=0)
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()
                    _, _, _, _, _, _, _, _, img_all, text_all, _, _, _, _, _ = model(imgs, txts)
                    t_imgs.append(img_all.cpu().numpy())
                    t_txts.append(text_all.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_imgs = np.concatenate(t_imgs)
            t_txts = np.concatenate(t_txts)
            t_labels = np.concatenate(t_labels).argmax(1)


            img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
            txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

            if phase == 'train':
                print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))

            if phase == 'train' and (img2text + txt2img) / 2. > best_acc:
                torch.cuda.empty_cache()
                best_acc = (img2text + txt2img) / 2.
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

    print(best_epoch)
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model_gat_loss weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "model/"+dataset+"_pre-model_gat_loss.pt")
    return model


def train_gan(model_genT, model_disT, model_gen, model_dis, model, data_loaders, optimizer_genT, optimizer_disT,
                optimizer_gen, optimizer_dis, dataset, batch_size, num_epochsGAN=100):

    best_model_wts_adv = copy.deepcopy(model_gen.state_dict())
    best_model_wts_advT = copy.deepcopy(model_genT.state_dict())
    best_acc_adv_text = 0.0
    best_acc_adv_image = 0.0
    best_epoch = 0
    model.eval()
    idx = 0
    for epoch in range(num_epochsGAN):
        print('Epoch {}/{}'.format(epoch + 1, num_epochsGAN))
        print('-' * 20)
        idx = idx + 1
        for phase in ['train', 'test']:
            if phase == 'train':
                model_genT.train()
                model_gen.train()
                model_disT.train()
                model_dis.train()
            else:
                model_genT.eval()
                model_gen.eval()
                model_disT.eval()
                model_dis.eval()

            running_loss_g = 0.0
            running_loss_d = 0.0
            running_loss_gT = 0.0
            running_loss_dT = 0.0


            # Iterate over data.
            for imgs, txts, labels in data_loaders[phase]:
                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")
                if imgs.shape[0] < batch_size:
                    cha = batch_size - imgs.shape[0]
                    # print(cha)
                    # print(imgs.shape[0] - cha)
                    imgs = torch.cat([imgs, imgs[imgs.shape[0] - cha:].clone()], dim=0)
                    txts = torch.cat([txts, txts[txts.shape[0] - cha:].clone()], dim=0)
                    labels = torch.cat([labels, labels[labels.shape[0] - cha:].clone()], dim=0)

                with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()

                    _, _, _, _, _, _, _, _, constraint_1, constraint_2, _, _, _, _, _ = model(imgs, txts)
                    if True:
                        optimizer_dis.zero_grad()

                        genimg = model_gen(imgs)
                        score_f = model_dis(genimg)
                        gentxt = model_genT(txts)
                        score_r = model_dis(gentxt)

                        dloss = - (torch.log(score_r) + torch.log(1 - score_f))
                        dloss = dloss.sum().mean()

                        if phase == 'train':
                            dloss.backward()
                            running_loss_d += dloss.item()
                            optimizer_dis.step()

                    if True:
                        optimizer_gen.zero_grad()
                        genimg = model_gen(imgs)
                        score_f = model_dis(genimg)

                        predloss = calc_loss2(genimg, constraint_1.detach(), model.shareClassifier(genimg), labels, 3)

                        gloss = -torch.log(score_f) + 1.2 * torch.log(predloss)
                        gloss = gloss.sum().mean()


                        if phase == 'train':
                            gloss.backward()
                            running_loss_g += gloss.item()
                            optimizer_gen.step()

                    ######################
                    if True:
                        optimizer_disT.zero_grad()

                        gentxt = model_genT(txts)
                        score_f = model_disT(gentxt)
                        genimg = model_gen(imgs)
                        score_r = model_disT(genimg)

                        dlossT = - (torch.log(score_r) + torch.log(1 - score_f))
                        dlossT = dlossT.sum().mean()

                        if phase == 'train':
                            dlossT.backward()
                            running_loss_dT += dlossT.item()
                            optimizer_disT.step()

                    if True:
                        optimizer_genT.zero_grad()

                        gentxt = model_genT(txts)
                        score_f = model_disT(gentxt)

                        predloss = calc_loss2(gentxt, constraint_2.detach(), model.shareClassifier(gentxt), labels, 12)

                        glossT = -torch.log(score_f) + 1.0 * torch.log(predloss)
                        glossT = glossT.sum().mean()

                        if phase == 'train':
                            glossT.backward()
                            running_loss_gT += glossT.item()
                            optimizer_genT.step()

            epoch_loss_g = running_loss_g / len(data_loaders[phase].dataset)
            epoch_loss_d = running_loss_d / len(data_loaders[phase].dataset)
            epoch_loss_gT = running_loss_gT / len(data_loaders[phase].dataset)
            epoch_loss_dT = running_loss_dT / len(data_loaders[phase].dataset)

            t_imgs, t_txts, t_labels = [], [], []
            t_imgs_adv = []
            t_txts_adv = []
            with torch.no_grad():
                for imgs, txts, labels in data_loaders['test']:
                    if imgs.shape[0] < batch_size:
                        cha = batch_size - imgs.shape[0]
                        imgs = torch.cat([imgs, imgs[imgs.shape[0] - cha:].clone()], dim=0)
                        txts = torch.cat([txts, txts[txts.shape[0] - cha:].clone()], dim=0)
                        labels = torch.cat([labels, labels[labels.shape[0] - cha:].clone()], dim=0)

                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()

                    _, _, _, _, _, _, _, _, constraint_1, constraint_2, _, _, _, _, _ = model(imgs, txts)
                    t_imgs.append(constraint_1.cpu().numpy())
                    t_txts.append(constraint_2.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
                    t_imgs_adv.append(model_gen(imgs).cpu().numpy())
                    t_txts_adv.append(model_genT(txts).cpu().numpy())
            t_imgs = np.concatenate(t_imgs)
            t_txts = np.concatenate(t_txts)

            t_imgs_adv = np.concatenate(t_imgs_adv)
            t_txts_adv = np.concatenate(t_txts_adv)

            t_labels = np.concatenate(t_labels).argmax(1)

            img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
            txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

            img2text_adv = fx_calc_map_label(t_imgs_adv, t_txts_adv, t_labels)
            txt2img_adv = fx_calc_map_label(t_txts_adv, t_imgs_adv, t_labels)

            if phase =="train":
                print('epoch_loss_g----{} Loss_Image: {:.4f} Loss_Text: {:.4f} ori - Img2Txt: {:.4f}  Txt2Img: {:.4f}'
                      .format(phase, epoch_loss_g, epoch_loss_gT, img2text, txt2img))
                print('epoch_loss_d----{} Loss_Image: {:.4f} Loss_Text: {:.4f} adv - Img2Txt: {:.4f}  Txt2Img: {:.4f}'
                      .format(phase, epoch_loss_d, epoch_loss_dT, img2text_adv, txt2img_adv))

            if phase == 'train':
                if img2text_adv + txt2img_adv > best_acc_adv_image + best_acc_adv_text:
                    best_acc_adv_image = img2text_adv
                    best_model_wts_adv = copy.deepcopy(model_gen.state_dict())

                    best_acc_adv_text = txt2img_adv
                    best_model_wts_advT = copy.deepcopy(model_genT.state_dict())
                    best_epoch = epoch + 1

    best_acc_adv = (best_acc_adv_image + best_acc_adv_text) / 2.0
    print(best_epoch)
    print('Adv - Best average ACC: {:4f}'.format(best_acc_adv))

    # load best model_gat_loss weights
    model_gen.load_state_dict(best_model_wts_adv)
    model_genT.load_state_dict(best_model_wts_advT)

    torch.save(model_gen.state_dict(), "model/"+dataset+"_pre-model_gen.pt")
    torch.save(model_genT.state_dict(), "model/"+dataset+"_pre-model_genT.pt")

    return model_genT, model_gen

def train_model(model_genT, model_disT, model_gen, model_dis, model, data_loaders,
                optimizer_genT, optimizer_disT, optimizer_gen, optimizer_dis, batch_size, optimizer,
                dataset, num_epochs=100, num_epochsGAN=100):

    model = train_gse(model,batch_size, data_loaders, optimizer, dataset, num_epochs)

    model_genT, model_gen = train_gan(model_genT, model_disT, model_gen, model_dis, model, data_loaders,
                                      optimizer_genT, optimizer_disT, optimizer_gen, optimizer_dis, dataset, batch_size, num_epochsGAN)

    return model_genT, model_gen, model

