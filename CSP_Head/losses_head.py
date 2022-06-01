import torch as th
import torch.nn as nn
import numpy as np

"""
labels[0][:,0,:,:] #Gaussian Mask
labels[2][:,2,:,:] #Classification
labels[1][:,0,:,:] #log(height)
"""


def loss_center(pred, labels):
    """
    pred : center map output of the detection head
    """
    classification_loss = nn.functional.binary_cross_entropy(
        pred.float().squeeze(dim = 1), labels[2][:, 2, :, :].float(), reduction="none")

    #positives = labels[2][:, 2, :, :] == 1  # positives(i,j) == 1 if yij = 1
    #negatives = labels[2][:, 2, :, :] < 1  # negatives(i,j) == 1 if yij = 0

    positives = labels[2][:,2,:,:]
    negatives = labels[2][:,1,:,:] - labels[2][:,2,:,:]

    foreground_weight = positives * (1.0 - pred) ** 2.0

    background_weight = negatives * \
        ((1.0 - labels[0][:, 0, :, :])**4.0)*(pred ** 2.0)  # Mij GaussMask

    focal_weight = foreground_weight + background_weight

    assigned_boxes = th.sum(
        labels[2][:, 2, :, :], dim=(1, 2))   # = nb of objects
    K = th.max(th.stack((assigned_boxes, th.ones(assigned_boxes.shape)), dim=0), dim=0)[
        0]  # nb of people
 
    class_loss = th.sum(th.sum(focal_weight*classification_loss, dim=(1,2,3)) / K).unsqueeze(dim = 0)

    return class_loss #class_loss


def loss_scale(h_pred, labels):
    """"
    h_pred : height prediction
    """

    mask_object = labels[1][:,1,:,:]
    assigned_boxes = th.sum(labels[1][:, 1, :, :], dim=(1, 2))
    K = th.max(th.stack((assigned_boxes, th.ones(
        assigned_boxes.shape)), dim=0), dim=0)[0]
    l1 = nn.L1Loss(reduction='none')
    
    loss = mask_object*l1( labels[1][:,0,:,:]/ (labels[1][:,0,:,:]+1e-10), h_pred.squeeze(dim = 1)/(labels[1][:,0,:,:] + 1e-10) )

    loss = th.sum(th.sum(loss, dim = (1,2))/K)
    print(K)
    return loss


def loss_scale_l1(h_pred, labels):
    """"
    h_pred : height prediction
    """
    assigned_boxes = th.sum(labels[2][:, 2, :, :], dim=(1, 2))
    K = th.max(th.stack((assigned_boxes, th.ones(
        assigned_boxes.shape)), dim=0), dim=0)[0]
    return th.sum(th.sum(th.abs(labels[1][:, 0, :, :] - th.log(h_pred)), dim=(1, 2))/K)

    # return nn.mean(nn.abs(labels[1][:,0,:,:] - th.log(h_pred[:,:,:,1]), dim = (1,2))/K)
    # index of y_tru and y_pred not correct, must be the height.

def loss(prediction, labels):
    pred_center, pred_h = prediction
    loss = 0.05*loss_scale(pred_h, labels) + 0.01 * loss_center(pred_center, labels)
    print(loss)
    return loss


def gaussian_base(label):
    """
    compute 2D Gaussian mask G(.) coefficient Mij of the loss function
    useless --> already implemented in the dataloading file.
    """
    x, y = label[0, 0].shape
    col = np.arange(y).reshape(1, y) + np.zeros(x).reshape(x, 1)
    row = np.arange(x).reshape(x, 1) + np.zeros(y).reshape(1, y)
    label = np.array([[0, 0, 1], [0, 1, 0]])
    index = np.where(label == 1)
    K_NB = np.count_nonzero(label)
    K = []
    for i in range(K_NB):
        G = np.multiply(
            np.exp(-(row - index[0][i])**2/(2*1)), np.exp(-(col - index[1][i])**2/(2*1)))
        K.append(G)
    return np.maximum.reduce(K)




"""
def regr_offset(y_true, y_pred):
	absolute_loss = th.abs(y_true[:, :, :, :2] - y_pred[:, :, :, :])
	square_loss = 0.5 * (y_true[:, :, :, :2] - y_pred[:, :, :, :]) ** 2
	l1_loss = y_true[:, :, :, 2] * th.sum(th.where(th.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5), axis=-1)
	assigned_boxes = th.sum(y_true[:, :, :, 2])
	class_loss = 0.1*th.sum(l1_loss) / th.maximum(1.0, assigned_boxes)
	return class_loss
"""