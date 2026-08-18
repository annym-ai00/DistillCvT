import argparse
import os
import datetime
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.cvt import CvT
from dataset.datasets import DatasetLoader
from dataset.samplers import CategoriesSampler
from utils import seed_torch, set_gpu, ensure_path, Averager, count_acc, euclidean_metric, Timer, compute_confidence_interval

def get_dataset(args):
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
        
    trainset = DatasetLoader(dataset_name=args.dataset, phase='train', size=args.image_size)
    valset = DatasetLoader(dataset_name=args.dataset, phase='valid', size=args.image_size)
    testset = DatasetLoader(dataset_name=args.dataset, phase='test', size=args.image_size)
    
    train_sampler = CategoriesSampler(trainset.label, args.train_batch,
                                        args.train_way, args.shot + args.train_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=args.worker, pin_memory=True)

    val_sampler = CategoriesSampler(valset.label, args.valid_batch,
                                    args.train_way, args.shot + args.train_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)
    
    test_sampler = CategoriesSampler(testset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=args.worker, pin_memory=True)
    
    return train_loader, val_loader, test_loader, n_cls

def main(args):
    ensure_path(args.save_path)

    train_loader, val_loader, test_loader, n_cls = get_dataset(args)
   
    model = CvT(num_classes=n_cls).cuda()
       
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1, verbose=True)
    
    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    def save_checkpoint(points, path, name='checkpoint'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(points, os.path.join(path, '{}.pth.tar'.format(name)))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_epoch'] = 0
    
    timer = Timer()
    best_epoch = 0
    start_epoch = 1
      
    # check resume point
    checkpoint_file = os.path.join(args.checkpoint_path, 'checkpoint.pth.tar')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        trlog = checkpoint['trlog']
        start_epoch = checkpoint['start_epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        trlog['max_acc'] = checkpoint['best_acc']
        trlog['max_epoch'] = checkpoint['best_epoch']
        print("=> Resume from epoch {} ...".format(start_epoch))
    
    
    for epoch in range(start_epoch, args.max_epoch + 1):

        tl, ta = train(args, model, train_loader, optimizer)
        lr_scheduler.step()
        vl, va, aa, bb = validate(args, model, val_loader)

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')
            trlog['max_epoch'] = epoch
            best_epoch = epoch
            cmi[0] = aa
            cmi[1] = bb
            
            # save best model
            save_checkpoint({
                'best_epoch': epoch,
                'model': model.state_dict()
            }, args.save_path, name='max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        
        # checkpoint saving
        save_checkpoint({
            'start_epoch': epoch,
            'best_acc': trlog['max_acc'],
            'best_epoch': trlog['max_epoch'],
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'trlog': trlog
        }, args.save_path)
        
        save_model('epoch-last')
        ot, ots = timer.measure()
        tt, _ = timer.measure(epoch / args.max_epoch)
        
        print('Epoch {}/{}, train loss={:.4f} - acc={:.4f} - val loss={:.4f} - acc={:.4f} - max acc={:.4f} - ETA:{}/{}'.format(
            epoch, args.max_epoch, tl, ta, vl, va, trlog['max_acc'], ots, timer.tts(tt-ot)))
        print("Best Epoch is {} with acc={:.2f}±{:.2f}%...".format(best_epoch, cmi[0], cmi[1]))
        
        print("------------------------------------------------------\n")

def train(args, model, train_loader, optimizer):
    model.train()

    tl = Averager()
    ta = Averager()

    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        
        p = args.shot * args.train_way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)  # (30, 1600)
        proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
        query = model(data_query)

        label = torch.arange(args.train_way).repeat(args.train_query)
        label = label.type(torch.cuda.LongTensor)

        logits = euclidean_metric(query, proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        proto = None; query = None; logits = None; loss = None

    return tl.item(), ta.item()

def validate(args, model, val_loader):
    model.eval()

    vl = Averager()
    va = Averager()
    acc_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)  # (30, 1600)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
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
    return vl.item(), va.item(), a, b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--checkpoint-path', default='')
    parser.add_argument('--save-path', default='')
    parser.add_argument('--gpu', default='0')
    # few-shot setting
    parser.add_argument('--shot', type=int, default=1) 
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    # dataset
    parser.add_argument('--dataset', type=str, default='cub', choices=['cub','dog','car'])
    parser.add_argument('--image-size', type=int, default=84)
    # network
    parser.add_argument('--worker', type=int, default=2)
    parser.add_argument('--max-epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--step-size', type=int, default=10)
    parser.add_argument('--train-batch', type=int, default=10000) #every epoch consist 1000 episode, total 30 epoch, so 30000 episode
    parser.add_argument('--valid-batch', type=int, default=1000)
    parser.add_argument('--test-batch', type=int, default=1000)
    args, _ = parser.parse_known_args()
    
    # Create the directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    start_time = datetime.datetime.now()

    # fix seed
    seed_torch(1)
    set_gpu(args.gpu)
    main(args)

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)
