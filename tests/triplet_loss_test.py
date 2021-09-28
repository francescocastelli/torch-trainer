import unittest
import torch
from torchtrainer.utils.losses import batch_hard_triplet_loss, _get_distance_matrix

class TripletLossTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_distances_not_squared(self):
       anchor = torch.rand(1, 128) 
       positive = torch.rand(1, 128)
       negative = torch.rand(1, 128)

       batch = torch.vstack((anchor, positive, negative))
       distances = _get_distance_matrix(batch, squareRoot=True)

       d_ap_my = distances[0, 1]
       d_an_my = distances[0, 2]

       d_ap_true = torch.cdist(anchor , positive)
       d_an_true = torch.cdist(anchor , negative)
    
       self.assertAlmostEqual(d_ap_my.item(), d_ap_true.item(), delta=0.01)
       self.assertAlmostEqual(d_an_my.item(), d_an_true.item(), delta=0.01)


    def test_distances_squared(self):
       anchor = torch.rand(1, 128) 
       positive = torch.rand(1, 128)
       negative = torch.rand(1, 128)

       batch = torch.vstack((anchor, positive, negative))
       distances = _get_distance_matrix(batch, squareRoot=False)

       d_ap_my = distances[0, 1]
       d_an_my = distances[0, 2]

       d_ap_true = torch.cdist(anchor , positive) ** 2
       d_an_true = torch.cdist(anchor , negative) ** 2
    
       self.assertAlmostEqual(d_ap_my.item(), d_ap_true.item(), delta=0.01)
       self.assertAlmostEqual(d_an_my.item(), d_an_true.item(), delta=0.01)


    # loss is zero when d(a, p) - d(a, n) + margin < 0
    # here we test the case: d(a, p) = 0, d(a, n) > margin ----> loss = 0
    def test_loss_zero_val_dap(self):
       anchor = torch.ones(1, 128) 
       positive = torch.ones(1, 128)
       negative = torch.zeros(1, 128)

       batch = torch.vstack((anchor, positive, negative))
       labels = torch.tensor([1.0, 1.0, 0.0])

       true_loss = torch.nn.TripletMarginLoss(margin=0.1, p=2.0)

       true_val = true_loss(anchor, positive, negative)
       my_val, _, _ = batch_hard_triplet_loss(batch, labels, 
                                              squareRoot=True, margin=0.1, 
                                              use_systems=False, use_speakers=False, 
                                              speakers=None, systems=None)

       self.assertAlmostEqual(true_val.item(), my_val.item(), delta=0.0)
       self.assertTrue(my_val.item() == 0)


    # loss is zero when d(a, p) - d(a, n) + margin < 0
    # here we test the case: d(a, p) = 0, d(a, n) = margin ----> loss = 0
    def test_loss_zero_val_dan_equal_margin(self):
       anchor = torch.ones(1, 128) 
       positive = torch.ones(1, 128)
       negative = torch.zeros(1, 128)

       batch = torch.vstack((anchor, positive, negative))
       labels = torch.tensor([1.0, 1.0, 0.0])

       true_loss = torch.nn.TripletMarginLoss(margin=0.1, p=2.0)

       true_val = true_loss(anchor, positive, negative)
       my_val, _, _ = batch_hard_triplet_loss(batch, labels, 
                                              squareRoot=True, margin=1.0, 
                                              use_systems=False, use_speakers=False, 
                                              speakers=None, systems=None)

       self.assertAlmostEqual(true_val.item(), my_val.item(), delta=0.0)
       self.assertTrue(my_val.item() == 0)


    # loss is equal to the margin when d(a, p)=0 and d(a, n)=0
    def test_loss_exactly_margin(self):
       anchor = torch.ones(1, 128) 
       positive = torch.ones(1, 128)
       negative = torch.ones(1, 128)

       batch = torch.vstack((anchor, positive, negative))
       labels = torch.tensor([1.0, 1.0, 0.0])

       true_loss = torch.nn.TripletMarginLoss(margin=0.1, p=2.0)

       true_val = true_loss(anchor, positive, negative)
       my_val, _, _ = batch_hard_triplet_loss(batch, labels, 
                                              squareRoot=True, margin=0.1, 
                                              use_systems=False, use_speakers=False, 
                                              speakers=None, systems=None)

       self.assertAlmostEqual(true_val.item(), my_val.item(), delta=0.01)



    def test_loss_one_val(self):
       x1 = torch.rand(1, 1) 
       x2 = torch.rand(1, 1)
       x3 = torch.rand(1, 1)

       batch = torch.vstack((x1, x2, x3))
       labels = torch.tensor([1.0, 1.0, 0.0])

       true_loss = torch.nn.TripletMarginLoss(margin=0.1, p=2.0)

       # these are the various loss with x1 as anchor, x2 as anchor and x3 as anchor
       # x1 and x2 are part of the same class, while x3 is part of another
       x1_anch = true_loss(x1, x2, x3)
       x2_anch = true_loss(x2, x1, x3)
       # there are two negative for x3
       x3_anch = true_loss(x3, x3, x1)
       x3_anch2 = true_loss(x3, x3, x2)

       # avg of the losses in the batch
       true_val = (x1_anch + x2_anch + torch.max(x3_anch, x3_anch2)) / 3

       my_val, _, _ = batch_hard_triplet_loss(batch, labels, 
                                              squareRoot=True, margin=0.1, 
                                              use_systems=False, use_speakers=False, 
                                              speakers=None, systems=None)

       self.assertAlmostEqual(true_val.item(), my_val.item(), delta=0.1)
