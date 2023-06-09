from torch import nn
import torch
from torch import optim
import datetime
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
#%%
#数据预处理
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
    ])
transform_imp= transforms.Compose([transforms.RandomVerticalFlip(),
                                   transforms.RandomCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])
data_path = '../data-unversioned/p1ch7/'
train_data1 = datasets.CIFAR10(data_path, train=True, download=True,
                              transform=transform_imp)
train_data2 =datasets.CIFAR10(data_path, train=True, download=True,
                              transform=transform_train)
train_data =train_data2+train_data1
test_data = datasets.CIFAR10(data_path, train=False, download=True,
                             transform=transform_test)
train_loader = DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=0)
#%%
#神经网络
class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,use_conv_1_1=True):
        super(ResBlock,self).__init__()
        self.conv1 =nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.conv2 =nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding = 0)
        self.batch_norm =nn.BatchNorm2d(num_features=out_channel)
        self.use_conv_1_1 = use_conv_1_1
        self.relu = nn.ReLU()
    def forward(self,in_put):
        out = self.conv2(in_put)
        out =self.batch_norm(out)
        out = self.conv1(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        if self.use_conv_1_1:
            in_put = self.conv2(in_put)
        return out+in_put
class AttentionBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(AttentionBlock,self).__init__()
        self.avg_p = nn.AdaptiveAvgPool2d(1)
        self.max_p = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel,in_channel,kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,in_put):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_p(in_put))))
        max_out = self.conv2(self.relu(self.conv1(self.max_p(in_put))))
        out = self.sigmoid(avg_out+max_out)
        return out*in_put
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,64,11,stride=1,padding=5),
                                    AttentionBlock(64,32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Dropout2d(p=0.9),
                                    nn.Conv2d(64, 128, 7, stride=1, padding=3),
                                    AttentionBlock(128,64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    ResBlock(128,256),
                                    AttentionBlock(256,128),
                                    nn.MaxPool2d(2,2),
                                    ResBlock(256,512),
                                   # ResBlock(512,1024),
                                    nn.AvgPool2d(4,4))
        self.layer2 = nn.Sequential(nn.Dropout2d(p=0.5),
                                    nn.Linear(512,10))
    def forward(self, imgs):
        out = self.layer1(imgs)
        out =out.view(-1,512)
        out = self.layer2(out)
        return out
#%%
# 参数检查
model = Model()
numel_list = [p.numel() for p in model.parameters()]
print(sum(numel_list), numel_list)
#%%
#训练函数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model().to(device=device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
def show_effect(train,test,epoch):
    loss_list = []
    total_list =[]
    correct_list = []
    name_list = []
    len_list =[]
    for name,loader in [('train',train),('val',test)]:
        correct = 0
        total = 0
        loss_t =0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                _, predicted = torch.max(outputs, dim=1)
                loss_t += loss.item()
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
            loss_list.append(loss_t)
            total_list.append(total)
            correct_list.append(correct)
            name_list.append(name)
            len_list.append(len(loader))
    print('{} Epoch {} |'.format(datetime.datetime.now(), epoch),end=' ')
    for i in range(2):
        print('{}_loss: {:.4f} Acc_{}: {:.2f}% |'.format(name_list[i],
                                                       loss_list[i] / len_list[i],
                                                       name_list[i],
                                                       100*correct_list[i]/total_list[i]),end=' ')
    print('')
def fit(n_epochs, optimizer_n, model_n, loss_f, train,test):
    for epoch in range(1,n_epochs+1):
        for imgs , labels in train:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model_n(imgs)
            loss = loss_f(outputs, labels)
            optimizer_n.zero_grad()
            loss.backward()
            optimizer_n.step()
        show_effect(train=train,test=test,epoch=epoch)
#%%
#主程序
fit(
n_epochs =100,
optimizer_n = optimizer,
  model_n = model,
loss_f = loss_fn,
train = train_loader,
test =test_loader)