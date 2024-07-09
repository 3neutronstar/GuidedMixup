import torch


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

def pearson_correlation(x:torch.tensor,y:torch.tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    mean_x=x.mean()
    mean_y=y.mean()

    r= torch.sum((x-mean_x)*(y-mean_y))/ ( (x-mean_x).norm() * (y-mean_y).norm() )

    return r

def cosine_similarity(a, b):
    dot = a.matmul(b.t())
    norm =a.norm(dim=1,keepdim=True).matmul(b.norm(dim=1,keepdim=True).t())
    return dot / norm
    