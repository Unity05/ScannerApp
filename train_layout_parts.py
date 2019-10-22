import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DenoisingAutoencoderLayoutParts
import data_generator
from multiprocessing.pool import ThreadPool


num_epochs = 250
batch_size = 1
batches_at_once = 5

iterations = int(num_epochs/batches_at_once)

pool = ThreadPool(processes=1)

model = DenoisingAutoencoderLayoutParts().cuda()
distance = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), weight_decay=5e-5, lr=0.0001)

batch_async_return = pool.apply_async(data_generator.get_text_batches,
                               kwds={'num_batches': batches_at_once, 'batch_size': batch_size,
                                      'salt': True, 'pepper': True})

for i in range(iterations):
    batches = batch_async_return.get()
    batch_async_return = pool.apply_async(data_generator.get_text_batches,
                                          kwds={'num_batches': batches_at_once, 'batch_size': batch_size,
                                                'salt': True, 'pepper': True})
    for epoch, batch in enumerate(batches):
        train, label = batch
        label = Variable(label).cuda()
        train = Variable(train).cuda()
        # ====forward====
        output = model(train)
        loss = distance(output, label)
        # ====backward====
        optim.zero_grad()
        loss.backward()
        optim.step()
        # ====log====
        print('epoch [{}/{}], loss:{:.4f}'.format((i)*batches_at_once+epoch, num_epochs, loss.item()))

torch.save(model, 'models/layout_parts_model_37.4.pth')
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            }, 'models/model_checkpoint_37.4')
print('##################################################################################\n'
      '############################# FINAL MODEL SAVED ##################################\n'
      '##################################################################################')
