''' 
implementation ported to pytorch from: 
    https://github.com/omoindrot/tensorflow-triplet-loss
'''

import torch 

#----------------------------------- utility functions ------------------------------------------
def _get_eq_mask(values):
    mask_same = torch.eq(torch.unsqueeze(values, 0), torch.unsqueeze(values, 1))
    return mask_same

def _get_positives_mask(labels, speakers, systems, use_speakers, use_systems):
    # get device from labels 
    device = labels.device
    #computes where labels are equal (tensor of shape (bs, bs), true iif labels[i] == labels[j])
    mask_eq = _get_eq_mask(labels)
    #computes where labels are not the same (tensor of shape (bs, bs), true iif i != j)
    mask_not_same = torch.logical_not(torch.eye(labels.shape[0], device=device))
    #final mask, shape of (bs, bs)
    mask_pos = torch.logical_and(mask_eq, mask_not_same)
    mask_pos = torch.logical_and(mask_pos, _get_eq_mask(speakers)) if use_speakers else mask_pos
    mask_pos = torch.logical_and(mask_pos, _get_eq_mask(systems)) if use_systems else mask_pos

    return mask_pos

def _get_negatives_mask(labels, speakers, use_speakers):
    #computes where labels are not equal (tensor of shape (bs, bs), true iif labels[i] != labels[j])
    mask_eq = _get_eq_mask(labels) 
    # mask[i, j] == True means that for anchor labels[i] the element labels[j] is a valid negative
    mask_neg = torch.logical_not(mask_eq)
    mask_neg = torch.logical_and(mask_neg, _get_eq_mask(speakers)) if use_speakers else mask_neg

    return mask_neg

def _get_valid_triplet_mask(labels, speakers, systems, use_speakers, use_systems):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
        - speakers[i] == speakers[j] == speakers[k]
        - systems[i] == systems[j]
    """
    # get device from labels 
    device = labels.device

    # Check that i, j and k are distinct
    indices_not_equal = torch.logical_not(torch.eye(labels.shape[0], device=device))
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k),
                                         j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = _get_eq_mask(labels) 
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))
    valid_mask = valid_labels

    # Check if speakers[i] == speakers[j] == speakers[k]
    if use_speakers:
        speakers_equal = _get_eq_mask(speakers)
        i_equal_j = torch.unsqueeze(speakers_equal, 2)
        i_equal_k = torch.unsqueeze(speakers_equal, 1)

        valid_speakers = torch.logical_and(i_equal_j, i_equal_k)
        valid_mask = torch.logical_and(valid_mask, valid_speakers)

    # Check if systems[i] == systems[j] 
    if use_systems:
        systems_equal = _get_eq_mask(systems)
        i_equal_j = torch.unsqueeze(systems_equal, 2)

        valid_mask = torch.logical_and(valid_mask, i_equal_j)

    # Combine the two masks
    mask = torch.logical_and(distinct_indices, valid_mask)
    return mask

def _get_distance_matrix(embeddings, squareRoot):
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))
    # Get squared L2 norm for each embedding
    square_norm = torch.diagonal(dot_product)

    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)
    distances = torch.clamp(distances, min=0.0)

    if squareRoot:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances

#------------------------------- loss functions -------------------------------------------------

def batch_all_triplet_loss(embeddings, labels, squared, speakers, systems, margin, args):
    # Get the pairwise distance matrix
    pairwise_dist = _get_distance_matrix(embeddings, squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    #assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)

    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    #assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_valid_triplet_mask(labels, speakers, systems, args.speakers, args.systems)
    mask = mask.float()
    triplet_loss = torch.mul(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.clamp(triplet_loss, min=0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.greater(triplet_loss, 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

def batch_hard_triplet_loss(embeddings, labels, squared, speakers, systems, margin, 
                            use_speakers, use_systems):

    # Get the pairwise distance matrix
    pairwise_dist = _get_distance_matrix(embeddings, squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_positives_mask(labels, speakers, systems, 
                                               use_speakers, use_systems)
    mask_anchor_positive = mask_anchor_positive.float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_negatives_mask(labels, speakers, use_speakers)
    mask_anchor_negative = mask_anchor_negative.float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, axis=1, keepdims=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss, hardest_positive_dist, hardest_negative_dist
