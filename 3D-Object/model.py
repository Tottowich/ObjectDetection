#from venv import main
#from re import T
import time
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"

class Object_Detector(nn.Module):
    def __init__(self,
                depth = 5,
                max_point = 1000,
                features = 100,
                input_features = 3,
                ):
        super(Object_Detector, self).__init__()

        self.linear = nn.Linear(input_features,features)
        self.seq = nn.Sequential(
            self.linear
            
        )
        
        
    def forward(self,x):
        return self.seq(x)
    

        


if __name__ == '__main__':
    print("Starting")


    model = Object_Detector(5,5,100).to(device)
    x = torch.Tensor(200,100,32,3).to(device)

    print(f"x Shape: {x.shape}")
    start = time.time()
    y = model(x)
    end = time.time()
    print(f"y Shape: {y.shape}")
    print(f"Time: {end-start}")
    print(f"GPU Available: {torch.cuda.is_available()}")










