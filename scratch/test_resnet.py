import torchvision.models.resnet as resnet
import torch.utils.model_zoo as model_zoo


temp = resnet.model_urls

for key in temp.keys():
    temp[key] = temp[key].replace("https://", "http://")

model1 = resnet.resnet50(pretrained=False)
model1.load_state_dict(model_zoo.load_url(resnet.model_urls["resnet50"]))

model2 = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3])
model2.load_state_dict(model_zoo.load_url(resnet.model_urls["resnet34"]))
