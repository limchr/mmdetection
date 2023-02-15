from train_dcgan import Discriminator, PrintLayer, unsupervised_ds
import torch
import torchvision.transforms as transforms



ngpu = 1
nc = 3
ndf = 64
#device = torch.device("cuda:0")
device = torch.device("cpu")

netD = Discriminator(ngpu, nc=nc, ndf=ndf).to(device)
netD.load_state_dict(torch.load('src/gan7/netD_epoch_50.pth'))


dataset = unsupervised_ds(root='/home/chris/data/okutama_action/converted_vanilla/',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
nc=3
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                    shuffle=True, num_workers=1)



with torch.enable_grad():
    for ni in dataloader:
        sample = ni[0].to(device)
        break


    # sampleclone = sample.clone().detach()
    sample.requires_grad = True
    sample.retain_grad()

    # result = netD.forward_reshape(sample)
    
    sr = torch.nn.functional.interpolate(sample,size=(64,64), mode='bilinear')
    # sr.requires_grad = True
    sr.retain_grad()

    result = netD.forward(sr)

    
    
    
    loss_component = torch.nn.MSELoss(reduction='mean')(result, torch.ones_like(result))

    loss_component.backward()

    print(sr.grad)



import numpy as np
import matplotlib.pyplot as plt
grads = sample.grad.squeeze()


grad = sri

g_std = torch.std(grad)
g_mean = torch.mean(grad)
grad = grad - g_mean
grad = grad / g_std

# Step 4: Update image using the calculated gradients (gradient ascent step)
img.data -= lr * grad



grads = grad.squeeze()
imsh = np.transpose(grads,(1,2,0))
plt.imshow(imsh)
plt.show()  