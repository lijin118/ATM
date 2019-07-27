import torch
import torch.nn as nn
import data.pre_process as prep
from torch.utils.data import DataLoader
from data.data_list import ImageList


def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False

                else:

                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False

                else:

                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

prep_dict = {}
prep_config = {"test_10crop": True, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
prep_dict["target"] = prep.image_train(**prep_config['params'])
if prep_config["test_10crop"]:
    prep_dict["test"] = prep.image_test_10crop(**prep_config['params'])
else:
    prep_dict["test"] = prep.image_test(**prep_config['params'])

dsets = {}
dset_loaders = {}
model_path = "model/d2a_74.298.pth"
test_path = 'data/amazon_list.txt'
if prep_config["test_10crop"]:
    for i in range(10):
        dsets["test"] = [ImageList(open(test_path).readlines(), \
                                   transform=prep_dict["test"][i]) for i in range(10)]
        dset_loaders["test"] = [DataLoader(dset, batch_size=4, \
                                           shuffle=False, num_workers=0) for dset in dsets['test']]
else:
    dsets["test"] = ImageList(open(test_path).readlines(), \
                              transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=4, \
                                      shuffle=False, num_workers=0)
model = torch.load(model_path)
model.eval()
print(image_classification_test(dset_loaders,model))

