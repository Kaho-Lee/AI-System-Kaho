# BSD 3-Clause License

# Copyright (c) 2017, 
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import KHlinear_cpp
import KHConv_cpp

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

class KHLinearCppFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights, bias=None):
        ctx.save_for_backward(input, weights, bias)

        if bias is not None:
            output = KHlinear_cpp.forward(input, weights, bias)
        else:
            output = KHlinear_cpp.forward(input, weights, torch.zeros(weights.size()[0]))

        return output[0]

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_input, grad_weight, grad_bias = KHlinear_cpp.backward(grad_output, input, weight, bias)
        return grad_input, grad_weight, grad_bias

class KHConvCppFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights, bias, stride, padding):

        output, col_input = KHConv_cpp.forward(input, weights, bias, stride, padding)
        ctx.save_for_backward(input, col_input, weights, bias, stride, padding)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, col_input, weights, bias, stride, padding = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_input, grad_weight, grad_bias = KHConv_cpp.backward(grad_output, input, col_input,\
            weights, bias, stride, padding)

        return grad_input, grad_weight, grad_bias,\
            torch.zeros_like(stride), torch.zeros_like(padding)

class KHCppLinear(nn.Module):

    def __init__(self, input_features, output_features, bias=True):
        super(KHCppLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)
            self.bias = None

        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
        
    def forward(self, input):
        return KHLinearCppFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias.shape if self.bias is not None else None
        )

class KHCppConv(nn.Module):

    def __init__(self, input_chanel, output_chanel, kernel_size, stride=1, padding=0):
        super(KHCppConv, self).__init__()
        self.input_chanel = input_chanel
        self.output_chanel = output_chanel
        if isinstance(kernel_size, int):
            self.kernel_height =  torch.tensor(kernel_size, requires_grad=False)
            self.kernel_width =  torch.tensor(kernel_size, requires_grad=False)
        elif isinstance(kernel_size, tuple):
            self.kernel_height =  torch.tensor(kernel_size[0], requires_grad=False)
            self.kernel_width =  torch.tensor(kernel_size[1], requires_grad=False)
        self.stride = torch.tensor(stride, requires_grad=False)
        self.padding = torch.tensor(padding, requires_grad=False)

        self.weight = nn.Parameter(torch.empty(self.output_chanel, self.input_chanel,\
            self.kernel_height, self.kernel_width))
        self.bias = nn.Parameter(torch.empty(self.output_chanel))

        self.weight.data.uniform_(-0.1, 0.1)
        self.bias.data.uniform_(-0.1, 0.1)
        
    def forward(self, input):
        return KHConvCppFunction.apply(input, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        return 'input_features={}, output_features={}, kernel_height={}, kernel_width={}, weight={} bias={} stride={} padding={}'.format(
            self.input_chanel, self.output_chanel, self.kernel_height, self.kernel_width, self.weight.shape, self.bias.shape, self.stride, self.padding
        )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = KHCppConv(1, 32, (3,3), 1)
        self.conv2 = KHCppConv(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = KHCppLinear(9216, 128)
        self.fc2 = KHCppLinear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

     # profiling model
    print("Start profiling...")
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        output = model(images[0].reshape(1,1,28,28))
    print(prof)

    print("Finished profiling.")

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
