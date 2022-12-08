[Tistory] https://it-the-hunter.tistory.com

[colab] https://colab.research.google.com/drive/1pcQ9Q4xxtVgLkooq6lz775ZUpw0QKYqI?hl=ko#scrollTo=CTx3jp-btolb

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
# colab-GPU 사용 확인
~~~py
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('GPU 연결 실패 !!')
else:
  print(gpu_info)
~~~
# RAM 사용량 체크
~~~py
ram_gb = virtual_memory().total / 1e9
print('{:.1f} gigabytes of available RAM\n'.format(ram_gb))
~~~
# pytorch-GPU 연결 확인
~~~py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('학습을 진행하는 기기:',device)
~~~
# 구글 드라이브 연결 
~~~py
from google.colab import drive
drive.mount('/content/drive')
~~~
1. fastai : 자연어처리, 추천 관련 모듈을 제공하며 딥러닝 진입장벽을 낮추기 위함
~~~py
!pip install fastai==2.4
from fastai.data.external import untar_data, URLs # 훈련된 데이터 또는 가중치를 사용하기 위함
~~~
2. 간단한 데이터셋 클래스 생성
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
3. 데이터를 하나씩 뽑기 위한 기본 문법
~~~py
x, y = next(iter(dataloader))
~~~
------------------------------------------------------------------------------------------------
