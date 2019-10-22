import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models import DenoisingAutoencoderSheetEdges
import data_generator
from multiprocessing.pool import ThreadPool


num_epochs = 2000
batch_size = 8
batches_at_once = 20

iterations = int(num_epochs/batches_at_once)

pool = ThreadPool(processes=1)

model = DenoisingAutoencoderSheetEdges().cuda()
distance = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), weight_decay=5e-5, lr=0.0001)

batch_async_return = pool.apply_async(data_generator.generate_images,
                               kwds={'width': 300, 'height': 400, 'number_of_batches': batches_at_once,
                                     'batch_size': batch_size,
                                     'noisy': False, 'color': False, 'save': True, 'return_batches': True,
                                     'destination_folder': 'image_tensors/'})

for i in range(iterations):
    used_batches, used_lines = batch_async_return.get()
    batch_async_return = pool.apply_async(data_generator.generate_images,
                                   kwds={'width': 300, 'height': 400, 'number_of_batches': batches_at_once,
                                         'batch_size': batch_size,
                                         'noisy': False, 'color': False, 'save': True, 'return_batches': True,
                                         'destination_folder': 'image_tensors/'})
    for epoch in range(batches_at_once):
        batch = used_batches[epoch]
        batch_tensor = Variable(batch).cuda()
        batch_tensor.requires_grad_(False)
        # ====forward====
        input_batch = data_generator.uncomplete_image(batch=batch_tensor, lines=used_lines[epoch], p=0.5)
        input_batch = data_generator.add_noise_bs(input_batch, p=0.2, color=False)
        output = model(input_batch)
        batch_tensor = Variable(batch).cuda()
        loss = distance(output, batch_tensor)
        # ====backward====
        optim.zero_grad()
        loss.backward()
        optim.step()
        # ====log====
        if loss.item() < 0.0001:
            print(output[0])
            print(batch_tensor[0])
            plt.imshow(batch_tensor[0].squeeze(0).cpu().detach())
            plt.show()
            plt.imshow(output[0].squeeze(0).cpu().detach())
            plt.show()
        if loss.item() <= 0.001:
            model.hard_mode = True

        print('epoch [{}/{}], loss:{:.4f}'.format((i)*batches_at_once+epoch, num_epochs, loss.item()))


torch.save(model, 'models/model_3.pth')
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            }, 'models/model_checkpoint_3')
print('##################################################################################\n'
      '############################# FINAL MODEL SAVED ##################################\n'
      '##################################################################################')
