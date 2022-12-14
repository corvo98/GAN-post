[Tistory] https://it-the-hunter.tistory.com
------------------------------------------------------------------------------------------------
2022.11.19 - Intro
- def : 함수_매개변수를 받아서 연산 후 결과를 리턴
- class : 객체를 생성하는 설계도
"엑셀 파일이 주어졌을 시, 읽어오는 코드"

- pandas 라이브러리로 읽어오기
~~~py
import pandas as pd
df = pd. read_execel('엑셀 파일 경로')
data = df.iloc[:, :-1]
label = df.iloc[-1]
~~~

- pytorh에 맞게 변환
~~~py
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset  #data loader, tensor dataset
df = pd.read_excel('엑셀 파일 경로')
data = df.iloc[:, :-1]
label = df.iloc[-1]
torch_data = torch.Tensor(data.to_numpy())
torch_label = torch.Tensor(label.to_numpy())
torch_whole_data = TensorDataset(torch_data, torch_label)
train_loader = Dataloader(torch_data, batch_size=16)
~~~

- KL Divergence : 두 확률분포의 차이를 계산하는데 사용되는 함수 <- 거리 개념이 아님

------------------------------------------------------------------------------------------------
2022.11.24 - DataLoader
- Image as Data
1. Max Pooling
2. Average Pooling
- DataLoader : 수십, 수백장의 데이터를 복잡하지 않고 쉽게 사용하기 위함
--parameters of DataLoader
1. dataset : 실제로 넣을 데이터 (tensor로 변환)
2. batch_size : GPU에 한번에 몇개의 이미지를 넣을지 그 배치 크기
3. shuffle : epoch마다 임의로 데이터가 섞이는 것을 의미 (True일 경우에만)
4. num_workers : 멀티 프로세싱 (일반적으로 GPUx2 or x4)
5. pin_memory : CPU의 일정한 램을 미리 할당하기 위함
6. drop_last : 배치로 나누다 남는 것들 <- 데이터 수를 맞추기 위함

[colab] https://colab.research.google.com/drive/1pcQ9Q4xxtVgLkooq6lz775ZUpw0QKYqI?hl=ko#scrollTo=03CF65dRzap2

- colab-GPU 사용 확인
~~~py
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('GPU 연결 실패 !!')
else:
  print(gpu_info)
~~~

- RAM 사용량 체크
~~~py
ram_gb = virtual_memory().total / 1e9
print('{:.1f} gigabytes of available RAM\n'.format(ram_gb))
~~~

- pytorch-GPU 연결 확인
~~~py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('학습을 진행하는 기기:',device)
~~~

- 구글 드라이브 연결 
~~~py
from google.colab import drive
drive.mount('/content/drive')
~~~

- fastai : 자연어처리, 추천 관련 모듈을 제공하며 딥러닝 진입장벽을 낮추기 위함
~~~py
!pip install fastai==2.4
from fastai.data.external import untar_data, URLs # 훈련된 데이터 또는 가중치를 사용하기 위함
~~~

- 간단한 데이터셋 클래스 생성
~~~py
class myDataset(Dataset):

  # 생성자 
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __getitem__(self, index):
    return self.x[index], self.y[index]

  def __len__(self):
    return self.x.shape[0]
~~~

- 데이터를 하나씩 뽑기 위한 기본 문법
~~~py
x, y = next(iter(dataloader))
~~~

------------------------------------------------------------------------------------------------
2022.11.30 - image to Tensor (by torch)
- PIL
[colab] https://colab.research.google.com/drive/18lva9pBppOgc-IHBF7PiOnG-iilwgRD-

~~~py
# pytorch를 이용해 이미지를 Tensor 형식으로 불러오기
import PIL 
import torchvision.transforms as transforms

# PIL -> Tensor
img = PIL.Image.open('/root/test_images/1091500_20221207030912_1.png')

tf = transforms.ToTensor()
img_t = tf(img) # Tensor로 변환

print(img_t.size())  # channel, height, width

# Tensor -> PIL
tf = transforms.ToPILImage()
img_t = tf(img_t)
print(img_t)

img_t
~~~

- matplotlib
[colab] https://colab.research.google.com/drive/1JtKLBIiKU1e85EBeTzh2wBTg0_ICURLt

~~~py
import torch
import PIL
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

img = PIL.Image.open('/content/1091500_20221208025259_1.png')
tf = transforms.ToTensor()
img_t = tf(img)

print(img_t.size())

img_t = img_t.permute(1, 2, 0)

print(img_t.size())

plt.imshow(img_t)
~~~

------------------------------------------------------------------------------------------------
2022.12.01 - image normalize and DataLoader
[colab] https://colab.research.google.com/drive/15ARAy0l_zr46N5hANAbpAkJKq_uVB9AW
~~~py
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms



trans = transforms.Compose([transforms.Resize((1000,563)),  # 이미지 resize 
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])  # 이미지 정규화
trainset = torchvision.datasets.ImageFolder(root = '/', transform = trans)

trainloader = DataLoader(trainset, batch_size=5, shuffle=False, num_workers=2)
dataiter = iter(trainloader)
images = dataiter.next()
print(images)
~~~

------------------------------------------------------------------------------------------------
2022.12.09 - dataloader
[colab] https://colab.research.google.com/drive/1raoUhYSxsGZ3MBqcevkXkmk8lrTwPMs9#scrollTo=CaHrlHrW7ttU

- coco data를 colab에서 바로 다운

~~~py
from fastai.data.external import untar_data, URLs
import glob

coco_path = untar_data(URLs.COCO_SAMPLE)
~~~

- coco image들의 경로를 모아 리스트로 작성

~~~py
paths = glob.glob(str(coco_path) + "/train_sample/*.jpg")
~~~

- 이미지들을 각각 train, validation 데이터로 사용

~~~py
import numpy as np

np.random.seed(1)
chosen_paths = np.random.choice(paths, 5000, replace=False)
index = np.random.permutation(5000)

train_paths = chosen_paths[index[:4000]]  # 앞의 4000장을 train 이미지로 사용
val_paths = chosen_paths[index[4000:]]  # 뒤의 1000장은 validaion 이미지로 사용

print(len(train_paths))
print(len(val_paths))
~~~

- 데이터 전처리 및 입력

~~~py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image # from cv2 import cv2
from skimage.color import rgb2lab, lab2rgb
import numpy as np

class ColorizationDataset(Dataset):
  # 생성자
  def __init__(self, paths, mode='train'):
    self.mode = mode
    self.paths = paths

    # if else문으로 mode가 train이 아닌지를 판별
    if mode == "train":     # train인 경우, 이미지 사이즈를 정규화하고 random horizontal flip으로 augmentation
      self.transforms = transforms.Compose([
          transforms.Resize((256, 256), Image.BICUBIC),
          transforms.RandomHorizontalFlip(),
      ])

    elif mode == "val":    # validation인 경우, 이미지 사이즈만 정규화
      self.transforms = transforms.Resize((256, 256), Image.BICUBIC)

    else:
      raise Exception("train or validation only !!!")

  def __getitem__(self, index):
    img = Image.open(self.paths[index]).convert("RGB")  # 이미지 불러오기
    img = np.array(self.transforms(img))  # 이미지를 transform에 넣고, 이미지를 numpy array로 변환
    img = rgb2lab(img).astype("float32") # 이미지를 lab 채널로 바꿈
    img = transforms.ToTensor()(img)  # 이미지를 tensor로 변환
    L = img[[0], ...] / 50. -1
    ab = img[[1, 2], ...] / 110.  # -1 -1 사이로 정규화 진행
    return {'L' : L, 'ab' : ab}

  def __len__(self):
    return len(self.paths)

dataset_train = ColorizationDataset(train_paths, mode='train')
dataset_val = ColorizationDataset(val_paths, mode='val')
~~~

- dataloader 생성 및 size 확인

~~~py
dataset_train = ColorizationDataset(train_paths, mode='train')
dataset_val = ColorizationDataset(val_paths, mode='val')

dataloader_train = DataLoader(dataset_train, batch_size=15, num_workers=3, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=15, num_workers=3, pin_memory=True)

data = next(iter(dataloader_train))
Ls, abs = data['L'], data['ab']
print(Ls.shape, abs.shape)
print(len(dataloader_train), len(dataloader_val))
~~~
