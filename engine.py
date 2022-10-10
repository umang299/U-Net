import torch
import config
from tqdm import tqdm


def train_fn(dataloader, model, optimizer, loss_fn, scaler):
    loop = tqdm(dataloader)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        print(f"{batch_idx} =>  Loss : {loss}")

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
