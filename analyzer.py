import torch
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.datasets import ImageFolder
from PIL import Image

class Analyzer():
    def __init__(self):
        model = inception_v3(num_classes=16)
        params = torch.load('params.pth', map_location='cpu')
        model.load_state_dict(params)
        self.model = model.cpu()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229,0.224,0.225])
        ])

        self.class_list = [
            'actinic keratosis', 
            'angioma', 
            'atypical melanocytic proliferation', 
            'basal cell carcinoma', 
            'dermatofibroma', 
            'lentigo NOS', 
            'lentigo simplex', 
            'melanoma', 
            'nevus', 
            'other', 
            'pigmented benign keratosis', 
            'seborrheic keratosis', 
            'solar lentigo', 
            'squamous cell carcinoma', 
            'vascular lesion'
        ]
        self.introduction = [
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
        ]

    def __call__(self, image):
        self.model.eval()
        img = Image.open(image)
        img = self.transform(img).unsqueeze(0)
        output = self.model(img)
        pred = output.max(1)[1][0]
        return (self.class_list[pred], self.introduction[pred])

    def pred(self, images):
        dataset = ImageFolder(images, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=50)
        for idx, (data, target) in enumerate(dataloader):
            if idx < 2:
                continue
            result = self.model(data).max(1)[1]
            print(target == result)
            print(target)
            break

if __name__ == '__main__':
    analyzer = Analyzer()
    analyzer.pred('/Users/icefires/Desktop/Test Examples')
