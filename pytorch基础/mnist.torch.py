import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# part 1: 数据集的获取，torch中提供了数据集的相关API
mnist_train_dataset = datasets.MNIST(root="./data/",
                                      train=True,
                                      download=True,
                                      transform=
                                        transforms.Compose([
                                                            transforms.Resize([28,28]),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.5],std=[0.5]), 
                                                            ])
                                    )
                                        
mnist_test_dataset = datasets.MNIST(root="./data/",
                                      train=False,
                                      download=True,
                                      transform = transforms.Compose([
                                                            transforms.Resize([28,28]),
                                                            transforms.ToTensor()
                                                            ])
                                    )

# part 2: 数据装载， dataloader
data_loader_train = torch.utils.data.DataLoader(
    dataset=mnist_train_dataset,
    batch_size=128,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=mnist_test_dataset,
    batch_size = 1,
    shuffle=True
)


# part 3: 数据可视化，检查数据
'''
images,labels = next(iter(data_loader_train))
# TypeError: img should be PIL Image. Got <class 'torch.Tensor'>
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
std=mean=[0.5,0.5,0.5]
img = img * std + mean
# 直接imshow会报错：Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
# 意思是需要归一化处理
print([int(labels[i].numpy()) for i,label in enumerate(labels)])
plt.imshow(img)
plt.show()
'''

# part 4: 模型搭建
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.features = nn.Sequential(
            # out = (in-kernelSize+2*padding)/stride+1
            nn.Conv2d(1,8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(8,16,kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            # out = out/2
            nn.MaxPool2d(stride=2,kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*128, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            torch.nn.Linear(1024, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 7*7*128)
        x = self.classifier(x)
        return x

# part 5: 训练以及测试
epochs=5
model=Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    print("-"*5,"train","-"*5)
    train_loss,train_correct = 0,0
    for data in data_loader_train:
        x_train, y_train = data
        x_train, y_train = Variable(x_train), Variable(y_train)

        optimizer.zero_grad()

        outputs = model(x_train)
        _,pred = torch.max(outputs.data, 1)

        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()

        # record loss and correct num
        train_loss += loss.data
        train_correct += torch.sum(pred==y_train.data)
    test_correct = 0

    print('-'*5,"test",'-'*5)
    for data in data_loader_test:
        x_test, y_test = data
        x_test, y_test = Variable(x_test), Variable(y_test)

        outputs = model(x_test)

        test_correct += torch.sum(pred==y_test.data)
    
    print("train loss: %.3f \t train acc: %.3f%%\t test acc: %.3f%%" % (
        train_loss,
        train_correct/len(data_loader_train),
        test_correct/len(data_loader_test)))