import torch
import torch.nn.functional as F


class NegativeCosineSimilarity(torch.nn.Module):
    def __init__(self, stop_gradient=True, *args, **kwargs):
        """
        Negative cosine similarity loss.

        :param stop_gradient: bool. If True, the second argument in the forward call will be detached before
                              calculating the loss. This is used for contrastive learning without negative samples,
                              like BYOL and SimSiam.
        """
        super().__init__(*args, **kwargs)
        self.stop_gradient = stop_gradient

    def forward(self, p, z):
        """
        The forward call.

        :param p: torch.tensor. Tensor of shape (n_batch, n_features).
        :param z: torch.tensor. Tensor of shape (n_batch, n_features).
        :return:
        """
        if self.stop_gradient:
            z = z.detach()
        p = p / torch.norm(p, dim=1, keepdim=True)
        z = z / torch.norm(z, dim=1, keepdim=True)
        return -(p * z).sum(dim=1).mean()


class SymmetricNegativeCosineSimilarity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Symmetric negative cosine similarity loss that measures the similarity between p1-z2 and p2-z1.
        """
        super().__init__(*args, **kwargs)
        self.ncl = NegativeCosineSimilarity(stop_gradient=True)

    def forward(self, p1, p2, z1, z2):
        l = self.ncl(p1, z2) / 2 + self.ncl(p2, z1) / 2
        return l


class MarginLoss(torch.nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, negative_weight=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.negative_weight = negative_weight

    def forward(self, x, labels):
        batch_size = x.size(0)

        # For positive class: don't care about it if prediction is already larger than m_plus
        left = F.relu(self.m_plus - x)
        # For negative class: don't care about it if prediction is already smaller than m_minus
        right = F.relu(x - self.m_minus)

        loss = labels * left + self.negative_weight * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss


class TotalVariationLoss(torch.nn.Module):
    def __init__(self, weight=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def forward(self, *images, **image_dict):
        image_list = []
        if len(images) > 0:
            image_list = image_list + list(images)
        if len(image_dict) > 0:
            image_list = image_list + list(image_dict.values())
        tv_loss = 0.0
        for arr in image_list:
            tv = torch.mean(torch.abs(arr[:, :, 1:, :] - arr[:, :, :-1, :]), dim=(-1, -2))
            tv = tv + torch.mean(torch.abs(arr[:, :, :, 1:] - arr[:, :, :, :-1]), dim=(-1, -2))
            tv = torch.mean(tv)
            tv_loss = tv_loss + tv
        tv_loss = self.weight * (tv_loss / len(image_list))
        return tv_loss
