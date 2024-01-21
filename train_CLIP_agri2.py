from torch.utils.data import Dataset, DataLoader
import torch
import clip
from torch import nn, optim
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from collections import OrderedDict
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
class ImageDataset(Dataset):
    def __init__(self, image_files, captions, preprocess=None):
        self.image_files = image_files
        self.captions = captions
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess = preprocess
        if preprocess is None:
            pass
        else:
            self.preprocess = preprocess

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        image = self.preprocess(image)
        # image_description = clip.tokenize(["caption"])
        caption = self.captions[idx]
        return image,  caption


def get_files(input_dir):
    jpg_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))

    # 对 JPG 文件列表按名称排序
    jpg_files = sorted(jpg_files)

    # 构建 JPG 文件的完整路径列表
    image_files = [file for file in jpg_files]

    captions = [file.split('/')[-2] for file in image_files]
    # for i in range(len(captions)):
    #     captions[i] = "a photo of " + captions[i].replace('_', ' ')

    return image_files,captions

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def load_pretrian_model(model_path):
    model, preprocess = clip.load(model_path, device=device, jit=False)  # 训练时 jit必须设置为false
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    return model, preprocess

def train(model,epoch, batch_size, learning_rate, train_dataloader,captions):
    # 加载模型

    #设置参数
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    for i in range(epoch):
        for j,batch in enumerate(train_dataloader):
            list_image, list_txt = batch  # list_images is list of image in numpy array(np.uint8), or list of PIL images

            if len(list_txt) < batch_size:
                break


            texts = clip.tokenize(list_txt).to(device)
            images = list_image.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            if device == "cpu":
                ground_truth = torch.arange(batch_size).long().to(device)
            else:
                #ground_truth = torch.arange(batch_size).half().to(device)
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)


            #反向传播
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            optimizer.zero_grad()
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            print('epoch[%d] batch[%d] loss: %.3f' % (i+1,j + 1, total_loss))

        print('[%d] loss: %.3f' %(i + 1,total_loss))

    torch.save(model.state_dict(), '/d/mqy/model/CLIP_agri_test.pt')

def eval_on_dataset( model,test_dataloader,captions):

    batch_size = 32

    captions_unique = list(OrderedDict.fromkeys(captions))
    #print(captions_unique)
    text_inputs = torch.cat([clip.tokenize(captions_unique[i]) for i in range(len(captions_unique))]).to(device)
    text_inputs = text_inputs.to(device)


    class_to_number = {class_name: idx for idx, class_name in enumerate(captions_unique)}

    model.eval()

    acc, acc_baseline, total = 0, 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for i, batch in enumerate(test_dataloader):

            list_image, list_txt = batch
            if len(list_txt) < batch_size:
                break

            image = list_image.to(device)


            mapped_numbers = [class_to_number[cls] for cls in list_txt]
            target = torch.tensor(mapped_numbers).to(device)

            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(target).sum().item()
            total += target.size(0)


    return acc_baseline / total

def main():
    epoch = 3
    batch_size = 128
    learning_rate = 5e-5

    model, preprocess = clip.load("ViT-B/32", device=device)
    # checkpoint = torch.load("_3.pt")
    # model.load_state_dict(checkpoint)

    image_files_train,captions_train = get_files('/s/data_m/Plant_Data/Plant_Data/train')
    train_dataset = ImageDataset(image_files_train,
                                 captions_train,
                                 preprocess=preprocess,
                                 )
    print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)


    image_files_test, captions_test = get_files('/s/data_m/Plant_Data/Plant_Data/test')
    captions_unique = list(OrderedDict.fromkeys(captions_test))
    print(captions_unique)

    test_dataset = ImageDataset(image_files_test,
                                captions_test,
                                preprocess=preprocess,
                                )
    print(len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)



    train(model,epoch, batch_size, learning_rate, train_dataloader,captions_test)
    print("trained")
    acc = eval_on_dataset(model,test_dataloader,captions_test)
    formatted_accuracy = "{:.2f}".format(acc * 100)  # 使用格式化字符串保留两位小数
    print(f"{formatted_accuracy}%")
    torch.save(model.state_dict(), f'/d/mqy/model/CLIP_agri_gpu.pt')

if __name__ == '__main__':
    main()
