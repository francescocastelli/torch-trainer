import unittest
import torch
from torchtrainer.utils.metrics import running_balanced_accuracy, sensitivity, specificity
from sklearn.metrics import balanced_accuracy_score

class BalancedAccTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_sensitivity_worst(self):
        predictions = torch.ones((1, 30))
        labels = torch.zeros((1, 30))
        tpr = sensitivity(predictions, labels) 
        self.assertEqual(tpr.item(), 0.0)

    def test_sensitivity_best(self):
        predictions = torch.ones((1, 30))
        labels = torch.ones((1, 30))
        tpr = sensitivity(predictions, labels) 
        self.assertEqual(tpr.item(), 1.0)

    def test_specificity_worst(self):
        predictions = torch.zeros((1, 30))
        labels = torch.ones((1, 30))
        tnr = specificity(predictions, labels) 
        self.assertEqual(tnr.item(), 0.0)

    def test_specificity_best(self):
        predictions = torch.zeros((1, 30))
        labels = torch.zeros((1, 30))
        tnr = specificity(predictions, labels) 
        self.assertEqual(tnr.item(), 1.0)

    def test_balanced_accuracy_score(self):
        predictions = torch.tensor([0, 1, 0, 0, 1, 0])
        labels = torch.tensor([0, 1, 0, 0, 0, 1])

        sk_acc = balanced_accuracy_score(labels, predictions)
        acc = running_balanced_accuracy(predictions, labels)
        self.assertAlmostEqual(sk_acc, acc.item(), delta=0.01)

    def test_balanced_accuracy_best(self):
        predictions = torch.tensor([0, 1, 0, 0, 0, 1])
        labels = torch.tensor([0, 1, 0, 0, 0, 1])

        sk_acc = balanced_accuracy_score(labels, predictions)
        acc = running_balanced_accuracy(predictions, labels)
        self.assertAlmostEqual(sk_acc, acc.item(), delta=0.01)

    def test_balanced_accuracy_worst(self):
        predictions = torch.tensor([1, 0, 1, 1, 1, 0])
        labels = torch.tensor([0, 1, 0, 0, 0, 1])

        sk_acc = balanced_accuracy_score(labels, predictions)
        acc = running_balanced_accuracy(predictions, labels)
        self.assertAlmostEqual(sk_acc, acc.item(), delta=0.01)

if __name__ == '__main__':
    unittest.main()
