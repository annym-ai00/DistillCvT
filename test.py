import argparse
import os
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.cvt import CvT
from dataset.datasets import DatasetLoader
from dataset.samplers import CategoriesSampler
from utils import seed_torch, set_gpu, Averager, count_acc, euclidean_metric, Timer, compute_confidence_interval

def main(args):
    if args.dataset == 'cub':
        n_cls = 200
        print("=> CUB_200_2011...")
    elif args.dataset == 'dog':
        n_cls = 120
        print("=> Stanford_Dogs...")
    elif args.dataset == 'car':
        n_cls = 196
        print("=> Stanford_Cars...")
    else:
        print("Invalid dataset:", args.dataset)
        exit()

    testset = DatasetLoader(dataset_name=args.dataset, phase='test', size=args.image_size)

    test_sampler = CategoriesSampler(testset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=0, pin_memory=True)

    model = CvT(num_classes=n_cls).cuda()

    # check resume point
    checkpoint_file = os.path.join(args.save_path, 'max-acc.pth')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint)
        print("=> Test Accuracy...")

    model.eval()

    vl = Averager()
    va = Averager()
    acc_list = []
    
    for i, batch in enumerate(test_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.test_way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)  # (30, 1600)
        proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
        query = model(data_query)

        label = torch.arange(args.test_way).repeat(args.test_query)
        label = label.type(torch.cuda.LongTensor)

        logits = euclidean_metric(query, proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        vl.add(loss.item())
        va.add(acc)
        acc_list.append(acc*100)

        proto = None; query = None; logits = None; loss = None
    
    a,b = compute_confidence_interval(acc_list)
    print("Final accuracy with 95% interval : {:.2f}±{:.2f}".format(a, b))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=30)
    
    #5-way-1-shot setting
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--image-size', type=int, default=84)
    parser.add_argument('--test-batch', type=int, default=600)
    parser.add_argument('--save-path', default='')
    parser.add_argument('--dataset', type=str, default='cub', choices=['cub','dog','car'])
    args, _ = parser.parse_known_args()

    start_time = datetime.datetime.now()

    # fix seed
    seed_torch(1)
    set_gpu(args.gpu)
    main(args)

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)
