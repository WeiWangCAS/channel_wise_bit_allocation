import datetime
import torch
from torch.utils.data import DataLoader
import argparse

import dataloader
import net_resnet, net_vgg, net_alexnet
import csv

parser = argparse.ArgumentParser(description='channel-wise bit allocation for deep visual feature quantization')
parser.add_argument('--gpuid', default=0, type=int, help='gpu id')
parser.add_argument('--bits',default=2,type=int,help='bits')
parser.add_argument('--flag',default=0,type=int,help='0:calculate interval,1:calculate MSE')
parser.add_argument('--model', default='resnet18', type=str, help='model choose')
parser.add_argument('--quant_mode', default='uniform', type=str, help='quant mode choose')
parser.add_argument('--net_mode', default='resnet', type=str, help='net_mode')
parser.add_argument('--reserve_interval', default=10, type=int, help='rereserved interval')
parser.add_argument('--alpha', default=0.9, type=float, help='alpha')
parser.add_argument('--beta', default=0.1, type=float, help='beta')
parser.add_argument('--csv_path', default='./channel_allocation/', type=str, help='csv path')
parser.add_argument('--csv_name', default='resnet18', type=str, help='csv name')
args = parser.parse_args()

if __name__ == "__main__":
    if args.flag != 0 and args.flag != 1:
        print("please input flag index")
        print("0:calculate interval,1:calculate MSE")
    else:
        f = open('./result/mse.csv', 'w', encoding='utf-8')
        f_writer = csv.writer(f)
        if args.flag == 0:
            cal_bits = 7
        if args.flag == 1:
            cal_bits = 2
        for bits in range(cal_bits,8):
            begin = datetime.datetime.now()
            device = torch.device('cuda:' + str(args.gpuid) if torch.cuda.is_available() else 'cpu')
            if args.model == 'alexnet':
                model = net_alexnet.alexnet(bits, args.flag, args.gpuid, args.quant_mode, args.net_mode, args.reserve_interval, args.alpha, args.beta, args.csv_path, args.csv_name)
                model_weight_path = './alexnet.pth'
            if args.model == 'resnet18':
                model = net_resnet.resnet18(bits, args.flag, args.gpuid, args.quant_mode, args.net_mode, args.reserve_interval, args.alpha, args.beta, args.csv_path, args.csv_name)
                model_weight_path = './resnet18.pth'
            if args.model == 'vgg16':
                model = net_vgg.vgg16(bits, args.flag, args.gpuid, args.quant_mode, args.net_mode, args.reserve_interval, args.alpha, args.beta, args.csv_path, args.csv_name)
                model_weight_path = './vgg16.pth'
            model.load_state_dict(torch.load(model_weight_path))
            model.to(device)

            images_num = 2
            images, labels = dataloader.loadvaldata('./DataSet/', args.gpuid, images_num)
            print('image load over')
            loader = torch.utils.data.DataLoader(images, batch_size=1, num_workers=4)
            print('loader load over')

            count = 0
            mse = 0
            model.eval()
            with torch.no_grad():
                for img, index in iter(loader):
                    print(f'the current count is {count}')
                    print(f'The current label is {index.item()}')
                    img = img.to(device)
                    output = model(img)
                    if args.flag == 0:
                        print('cal interval')
                    if args.flag == 1:
                        mse += output
                        print(f'mse is {mse}')
                    count += 1
                    print('\n')

                if args.flag == 0:
                    print('cal interval over')
                if args.flag == 1:
                    average_mse = mse / images_num
                    print(f'The average mse is {average_mse}')

            end = datetime.datetime.now()
            print(f'The runing time is {(end - begin) / images_num}')

            f_writer.writerow([bits, mse / images_num])

            print(f'{bits}is over')