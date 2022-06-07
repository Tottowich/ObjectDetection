from model import Object_Detector
import wandb
import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
"""
Input: model: The model to train
       train_loader: The data loader for the training data
       optimizer: The optimizer to use, ADAM.
       learning_rate: The learning rate to use
       learning_rate_scheduler: The learning rate scheduler to use, One-cycle.
       criterion: The loss function to use.
       epochs: The number of epochs to train for.
Output: The trained model.
"""
def training_loop(model,
                  train_loader,
                  optimizer,
                  criterion,
                  learning_rate,
                  learning_rate_decay,
                  learning_rate_scheduluer,
                  epochs = 5,
                  ):
    """
    Basic training loop for the model
    """
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    return model