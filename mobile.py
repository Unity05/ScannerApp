import torch
import torch.quantization as quantization
from models import DenoisingAutoencoderLayoutParts


model = DenoisingAutoencoderLayoutParts()
model.load_state_dict(torch.load('models/model_checkpoint_38.0')["model_state_dict"])
model.eval()
qmodel = quantization.convert(module=model)
#example = torch.randn(1, 3, 1000, 1200)
#traced_script_module = torch.jit.trace(model, example)
#traced_script_module.save('mobile/android_model_0.pt')
qmodel = torch.jit.script(qmodel)
qmodel.save('mobile/android_model_0.pt')    # dir (here: mobile) must already excist, otherwise it throws runtime error
