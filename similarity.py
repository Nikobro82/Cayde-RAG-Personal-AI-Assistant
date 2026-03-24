import torch 



def get_scores(v1, v2, k):
    numerator = v1 @ v2.T
    similarities = numerator / (torch.norm(v1) * torch.norm(v2, dim=1))
    top_five_k = torch.topk(similarities, k)
    return top_five_k