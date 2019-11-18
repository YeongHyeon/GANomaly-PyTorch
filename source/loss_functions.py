import torch

def loss_enc(z_code, z_code_hat):

    l_enc = torch.sum((z_code - z_code_hat)**2, dim=(1))

    return l_enc

def loss_rec(x, x_hat):

    l_con = torch.sum(torch.abs(x - x_hat), dim=(1, 2, 3))

    return l_con

def loss_adv(dis_x, dis_x_hat, features_real, features_fake):

    l_adv = torch.sum((dis_x - dis_x_hat)**2, dim=(1))

    for fidx, _ in enumerate(features_real):
        feat_dim = len(features_real[fidx].shape)

        if(feat_dim == 4):
            l_adv += torch.sum((features_real[fidx] - features_fake[fidx])**2, dim=(1, 2, 3))
        elif(feat_dim == 3):
            l_adv += torch.sum((features_real[fidx] - features_fake[fidx])**2, dim=(1, 2))
        elif(feat_dim == 2):
            l_adv += torch.sum((features_real[fidx] - features_fake[fidx])**2, dim=(1))
        else:
            l_adv += torch.sum((features_real[fidx] - features_fake[fidx])**2)

    return l_adv

def loss_ganomaly(z_code, z_code_hat, x, x_hat, \
    dis_x, dis_x_hat, features_real, features_fake, \
    w_enc=1, w_con=50, w_adv=1):

    z_code, z_code_hat, x, x_hat, dis_x, dis_x_hat = \
        z_code.cpu(), z_code_hat.cpu(), x.cpu(), x_hat.cpu(), dis_x.cpu(), dis_x_hat.cpu()

    for fidx, _ in enumerate(features_real):
        features_real[fidx] = features_real[fidx].cpu()
        features_fake[fidx] = features_fake[fidx].cpu()

    l_enc = loss_enc(z_code, z_code_hat)
    l_con = loss_rec(x, x_hat)
    l_adv = loss_adv(dis_x, dis_x_hat, features_real, features_fake)

    l_tot = torch.mean((w_enc * l_enc) + (w_con * l_con) + (w_adv * l_adv))

    l_enc = torch.mean(l_enc)
    l_con = torch.mean(l_con)
    l_adv = torch.mean(l_adv)

    return l_tot, l_enc, l_con, l_adv

"""
# Loss 3: Adversarial loss (L2 distance)
loss_adv = tf.compat.v1.reduce_sum(tf.square(dis_x - dis_x_hat), axis=(1))
for fidx, _ in enumerate(features_real):
    feat_dim = len(features_real[fidx].shape)
    if(feat_dim == 4):
        loss_adv += tf.compat.v1.reduce_sum(tf.square(features_real[fidx] - features_fake[fidx]), axis=(1, 2, 3))
    elif(feat_dim == 3):
        loss_adv += tf.compat.v1.reduce_sum(tf.square(features_real[fidx] - features_fake[fidx]), axis=(1, 2))
    elif(feat_dim == 2):
        loss_adv += tf.compat.v1.reduce_sum(tf.square(features_real[fidx] - features_fake[fidx]), axis=(1))
    else:
        loss_adv += tf.compat.v1.reduce_sum(tf.square(features_real[fidx] - features_fake[fidx]))

mean_loss_enc = tf.compat.v1.reduce_mean(loss_enc)
mean_loss_con = tf.compat.v1.reduce_mean(loss_con)
mean_loss_adv = tf.compat.v1.reduce_mean(loss_adv)

loss = tf.compat.v1.reduce_mean((w_enc * loss_enc) + (w_con * loss_con) + (w_adv * loss_adv))
"""
