import torch
import torch.distributed as dist
from archs.fer import FER


if __name__ == '__main__':
    parser = FER.parser()
    opt = parser.parse_args()
    print(opt)

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(dist.get_rank())
    model = FER(opt)
    model.fit()
