import os
import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import time
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import  WeightedRandomSampler
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import csv
from cnn_finetune import make_model
from torch.autograd import Variable
import torch.nn.functional as F

 # 默认使用PIL读图
def default_loader(path):
    # return Image.open(path)
    return Image.open(path).convert('RGB')

# 训练集图片读取
class TrainDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row[0], row[2]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# 测试集图片读取
class TestDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row[0]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

# 数据增强：在给定角度中随机进行旋转
class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)

def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])

# 训练函数
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # 从训练集迭代器中获取训练数据
    for i, (images, target) in enumerate(train_loader):
        # 评估图片读取耗时
        data_time.update(time.time() - end)
        # 将图片和标签转化为tensor         
        image_var = torch.tensor(images).cuda(async=True)
        label = torch.tensor(target).cuda(async=True)

        # 将图片输入网络，前传，生成预测值
        y_pred = model(image_var)

        # 计算loss
        loss = criterion(y_pred, label)
        losses.update(loss.item(), images.size(0))

        # 计算top1正确率
        prec, PRED_COUNT = accuracy(y_pred.data, target, topk=(1, 1))
        acc.update(prec, PRED_COUNT)

        # 对梯度进行反向传播，使用随机梯度下降更新网络权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 评估训练耗时
        batch_time.update(time.time() - end)
        end = time.time()

        # 打印耗时与结果
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))

# 验证函数
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        image_var = torch.tensor(images).cuda(async=True)
        target = torch.tensor(labels).cuda(async=True)

        # 图片前传。验证和测试时不需要更新网络权重，所以使用torch.no_grad()，表示不计算梯度
        with torch.no_grad():
            y_pred = model(image_var)
            loss = criterion(y_pred, target)

        # measure accuracy and record loss
        prec, PRED_COUNT = accuracy(y_pred.data, labels, topk=(1, 1))
        losses.update(loss.item(), images.size(0))
        acc.update(prec, PRED_COUNT)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('TrainVal: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))

    print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
          ' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
    return acc.avg, losses.avg

# 测试函数
def test(test_loader, model):
    csv_map = OrderedDict({'filename': [], 'probability': []})
    # switch to evaluate mode
    model.eval()
    for i, (images, filepath) in enumerate(tqdm(test_loader)):
        # bs, ncrops, c, h, w = images.size()
        filepath = [os.path.basename(i) for i in filepath]
        image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

        with torch.no_grad():
            y_pred = model(image_var.type(torch.cuda.FloatTensor))
            # 使用softmax函数将图片预测结果转换成类别概率
            smax = nn.Softmax(1)
            smax_out = smax(y_pred)
            
        # 保存图片名称与预测概率
        csv_map['filename'].extend(filepath)
        for output in smax_out:
            prob = ';'.join([str(i) for i in output.data.tolist()])
            csv_map['probability'].append(prob)

    result = pd.DataFrame(csv_map)
    result['probability'] = result['probability'].map(lambda x: [float(i) for i in x.split(';')])
    result.to_csv('/content/drive/My Drive/tiangong/result/Mresnet/%s/probability.csv' % file_name, header=None, index=False)
    # 转换成提交样例中的格式
    sub_filename, sub_label = [],[]
    for index, row in result.iterrows():
        sub_filename.append(row['filename'])
        pred_label = np.argmax(row['probability'])
        if pred_label == 0:
            sub_label.append('OCEAN')
        if pred_label == 1:
            sub_label.append('MOUNTAIN')
        if pred_label == 2:
            sub_label.append('LAKE')
        if pred_label == 3:
            sub_label.append('FARMLAND')
        if pred_label == 4:
            sub_label.append('DESERT')
        if pred_label == 5:
            sub_label.append('CITY')
        
    # 生成结果文件，保存在result文件夹中，可用于直接提交
    submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
    submission.to_csv('/content/drive/My Drive/tiangong/result/Mresnet/%s/submission.csv' % file_name, header=None, index=False)
    return

# 用于计算精度和时间的变化
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 学习率衰减：lr = lr / lr_decay
def adjust_learning_rate():
    global lr
    lr = lr / lr_decay
    return optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)

# 计算top K准确率
def accuracy(y_pred, y_actual, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    final_acc = 0
    maxk = max(topk)
    # for prob_threshold in np.arange(0, 1, 0.01):
    PRED_COUNT = y_actual.size(0)
    PRED_CORRECT_COUNT = 0
    prob, pred = y_pred.topk(maxk, 1, True, True)
    # prob = np.where(prob > prob_threshold, prob, 0)
    for j in range(pred.size(0)):
        if int(y_actual[j]) == int(pred[j]):
            PRED_CORRECT_COUNT += 1
    if PRED_COUNT == 0:
        final_acc = 0
    else:
        final_acc = PRED_CORRECT_COUNT / PRED_COUNT
    return final_acc * 100, PRED_COUNT

import torch as th                                                                                                                                                                                                                                      
class cross_OHEM(nn.Module):                                                     
    """ Online hard example mining. 
    Needs input from nn.LogSotmax() """                                                                                                                             
    def __init__(self, ratio = 0.8, class_num = 6, alpha = None, gamma = 2, size_average=True):      
        super(cross_OHEM, self).__init__()     
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average                            
        self.ratio = ratio                                                         
                                                                                   
    def forward(self, x, y):                                                                                             
        num_inst = x.size(0)                                                       
        num_hns = int(self.ratio * num_inst)                                       
        x_ = x.clone()                                                             
        inst_losses = th.autograd.Variable(th.zeros(num_inst)).cuda()              
        for idx, label in enumerate(y.data):                                       
            inst_losses[idx] = -x_.data[idx, label]                                 
        #loss_incs = -x_.sum(1)                                                    
        _, idxs = inst_losses.topk(num_hns)                                        
        x_hn = x.index_select(0, idxs)                                             
        y_hn = y.index_select(0, idxs)  
        return th.nn.functional.cross_entropy(x_hn, y_hn)   
    
if __name__ == '__main__':
    # 随机种子
    randseed = 456
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    torch.cuda.manual_seed_all(randseed)
    random.seed(randseed)
    global file_name
    file_name = 'resnet152_456_cross_len'
    # 创建保存模型和结果的文件夹
    if not os.path.exists('/content/drive/My Drive/tiangong/model/%s' % file_name):
        os.makedirs('/content/drive/My Drive/tiangong/model/%s' % file_name)
    if not os.path.exists('/content/drive/My Drive/tiangong/result/Mresnet/%s' % file_name):
        os.makedirs('/content/drive/My Drive/tiangong/result/Mresnet/%s' % file_name)
    # 创建日志文件
    if not os.path.exists('/content/drive/My Drive/tiangong/result/Mresnet/%s.txt' % file_name):
        with open('/content/drive/My Drive/tiangong/result/Mresnet/%s.txt' % file_name, 'w') as acc_file:
            pass
    with open('/content/drive/My Drive/tiangong/result/Mresnet/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))

    label2num= {
        'OCEAN':0,
        'MOUNTAIN':1,
        'LAKE':2,
        'FARMLAND':3,
        'DESERT':4,
        'CITY':5
    }
    
    class_num =6
    border_name = "same"
    l2_lr = 0.01
    train_path = '/content/drive/My Drive/tiangong/train/'
    testA_path= '/content/drive/My Drive/tiangong/test/'
    testB_path= '/content/drive/My Drive/tiangong/testB/'
    workers = 0
    input_size = 256* 256
    learning_rate = 1e-3
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    batch_size = 30
    workers = 0
    stage_epochs = [25, 15, 10]  
    lr = 1e-4
    lr_decay = 5
    weight_decay = 1e-4
    stage = 0
    start_epoch = 0
    total_epochs = sum(stage_epochs)
    best_precision = 0
    lowest_loss = 100
    print_freq = 1
    evaluate = False
    resume = True
    #resume = False
    num_classes = 6
    val_ratio = 0.1
    
    train_csv = pd.read_csv('/content/drive/My Drive/tiangong/train_label0.csv',header = None,encoding='utf-8')
    train_csv[0] = train_csv[0].map(lambda x:train_path+x)
    ori_label = train_csv[1]
    train_csv[2] = train_csv[1].map(lambda x:label2num[x])
    train_csv = train_csv.drop([1], axis=1)
    
    test_csv = pd.read_csv('/content/drive/My Drive/tiangong/testA_99.7.csv',header = None,encoding='utf-8')
    test_csv[0] = test_csv[0].map(lambda x:testA_path+x)
    temp_label = test_csv[1]
    test_csv[2] = test_csv[1].map(lambda x:label2num[x])
    test_csv = test_csv.drop([1], axis=1)
    result = train_csv.append(test_csv)
    
    ocean = result[result[2] == 0]
    ocean = ocean.head(456) 
    
    mountain = result[result[2] == 1]
    mountain = mountain.head(463) 
    
    lake = result[result[2] == 2]
    farmland = result[result[2] == 3]
    
    desert = result[result[2] == 4]
    desert = desert.head(636)
    
    city = result[result[2] == 5]
    
    all_label = ocean.append(mountain)
    all_label = all_label.append(lake)
    all_label = all_label.append(farmland)
    all_label = all_label.append(desert)
    all_label = all_label.append(city)
    
    train_csv, val_csv = train_test_split(all_label, test_size = val_ratio, random_state=randseed, stratify = all_label[2])

    def make_classifier(in_features, num_classes):
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    model = make_model('resnet152', num_classes=6, pretrained=True, input_size=(256, 256), classifier_factory=make_classifier)
    model = model.cuda()

    if resume:
        checkpoint_path = '/content/drive/My Drive/tiangong/model/%s/checkpoint.pth.tar' % file_name
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            best_precision = checkpoint['best_precision']
            lowest_loss = checkpoint['lowest_loss']
            stage = checkpoint['stage']
            lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            # 如果中断点恰好为转换stage的点，需要特殊处理
            if start_epoch in np.cumsum(stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('/content/drive/My Drive/tiangong/model/%s/model_best.pth.tar' % file_name)['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    # 分离训练集和测试集，stratify参数用于分层抽样
#    train_data_list, val_data_list = train_test_split(all_data, test_size=val_ratio, random_state=666, stratify=all_data[3])
#    # 读取测试图片列表
    test_data_list = pd.read_csv('/content/drive/My Drive/tiangong/testB.csv',header = None,encoding='utf-8')
    test_data_list[0] = test_data_list[0].map(lambda x:testB_path+x)
    
    # 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 训练集图片变换，输入网络的尺寸为384*384
    train_data = TrainDataset(train_csv,
                              transform=transforms.Compose([
                                  transforms.Resize((256, 256)),
                                  transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  FixedRotation([0, 90, 180, 270]),
                                  #transforms.RandomCrop(224),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))

    # 验证集图片变换
    val_data = TrainDataset(val_csv,
                          transform=transforms.Compose([
                              transforms.Resize((256,256)),
                              #transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize,
                          ]))

    # 测试集图片变换
    test_data = TestDataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((256, 256)),
                               # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    # 生成图片迭代器
    weight_per_class = [0.044, 0.236, 0.2, 0.329, 0.065, 0.128]
    def make_weights_for_balanced_classes(train_data):                                                   
        weight = [0] * len(train_data)  
        i = 0                                            
        for images, target in enumerate(train_data):                                     
            weight[i] = weight_per_class[target[1]]     
            i = i+1                             
        return weight
    
    classesid = [0,1,2,3,4,5]
#    weight = [0.044, 0.236, 0.2, 0.329, 0.065, 0.128]
    weights = make_weights_for_balanced_classes(train_data)
    weights = torch.DoubleTensor(weights)  
    train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler = train_sampler)
#    val_sampler = WeightedRandomSampler(weight, num_samples = 400, replacement=True)
#    val_loader = DataLoader(val_data, batch_size=batch_size*2, sampler = val_sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size = batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)
    
    # 保存最新模型以及最优模型
    def save_checkpoint(state, is_best, is_lowest_loss, filename='/content/drive/My Drive/tiangong/model/%s/checkpoint.pth.tar' % file_name):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '/content/drive/My Drive/tiangong/model/%s/model_best.pth.tar' % file_name)
#        if is_lowest_loss:
#            shutil.copyfile(filename, '/content/drive/My Drive/tiangong/model/%s/lowest_loss.pth.tar' % file_name)
     # 保存最新模型以及最优模型
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = cross_OHEM().cuda()
    # 优化器，使用带amsgrad的Adam
    optimizer = optim.Adam(model.parameters(), lr, weight_decay = weight_decay, amsgrad=True)

    if evaluate:
        validate(val_loader, model, criterion)
    else:
        # 开始训练
        for epoch in range(start_epoch, total_epochs):
            train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            precision, avg_loss = validate(val_loader, model, criterion)

            # 在日志文件中记录每个epoch的精度和loss
            with open('/content/drive/My Drive/tiangong/result/Mresnet/%s.txt' % file_name, 'a') as acc_file:
                acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))

            # 记录最高精度与最低loss，保存最新模型与最佳模型
            # is_best 是bool 类型的数据,True或者False
            is_best = precision >= best_precision
            is_lowest_loss = avg_loss <= lowest_loss
            best_precision = max(precision, best_precision)
            lowest_loss = min(avg_loss, lowest_loss)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_precision': best_precision,
                'lowest_loss': lowest_loss,
                'stage': stage,
                'lr': lr,
            }
            save_checkpoint(state, is_best, is_lowest_loss)
            # 判断是否进行下一个stage
            if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('/content/drive/My Drive/tiangong/model/%s/model_best.pth.tar' % file_name)['state_dict'])
                print('Step into next stage')
                with open('/content/drive/My Drive/tiangong/result/Mresnet/%s.txt' % file_name, 'a') as acc_file:
                    acc_file.write('---------------Step into next stage----------------\n')
    # 记录线下最佳分数
    with open('/content/drive/My Drive/tiangong/result/Mresnet/%s.txt' % file_name, 'a') as acc_file:
       # acc_file.write('* best acc: %.8f  %s\n' % (best_precision, os.path.basename(__file__)))
        acc_file.write('* best acc: %.8f  %s\n' % (best_precision, file_name))
    with open('/content/drive/My Drive/tiangong/result/Mresnet/best_acc.txt', 'a') as acc_file:
        acc_file.write('%s  * best acc: %.8f  %s\n' % (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision, file_name))

    best_model = torch.load('/content/drive/My Drive/tiangong/model/%s/model_best.pth.tar' % file_name)
    model.load_state_dict(best_model['state_dict'])
    test(test_loader = test_loader, model=model)
    # 释放GPU缓存
#    if(os.path.exists('/content/drive/My Drive/tiangong/model/%s/model_best.pth.tar')):
#        os.remove('/content/drive/My Drive/tiangong/model/%s/continue_model_best.pth.tar')
#        print ('remove_test ：%s' %os.listdir('/content/drive/My Drive/tiangong/model/%s/continue_model_best.pth.tar'))
#    else:
#        print ("error！")
    torch.cuda.empty_cache()