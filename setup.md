## GPU Memory Usage

Delete model, optimizer, and scheduler to free up memory. For example, when you do a forward pass, you created the model, please delete it after the forward pass. Also when using LR finder, you need to create the model, optimizer and scheduler. Please delete them.

Sequence of events, This order of operations does the trick for me, removing the parameters and gradients from the GPU.

```python
delete model, optimizer and scheduler
gc.collect()
torch.cuda.empty_cache()
```

Source: https://github.com/huggingface/transformers/issues/1742
Source: https://discuss.pytorch.org/t/memory-leak-debugging-and-common-causes/67339 On how to debug using utils.show_gpu_usage()

## Weights & Biases

```bash
# CLI Interface
pip install wandb --upgrade # upgrade
wandb login --relogin # Login to wandb
# Subsequently, you can run your wandb inside your script.
# TODO: To explore other options like put the wandb keys in a config file.
```


## Prepare Data

```python
df_train, df_test, df_folds, sub = prepare.prepare_data()
```
Returns the dataframes for the training, testing, and folds data, and the submission dataframe.

[]: # Language: python
[]: # Path: prepare.py


## Plot Dataloader

Initialize the dataloader and plot random images.

```python
# Plot random images from dataloader for sanity check.
train_dataset = dataset.CustomDataset(
    df=df_folds,
    transforms=transformation.get_train_transforms(),
    mode="train",
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    # sampler=RandomSampler(dataset_train),
    **loader_params.train_loader,
    worker_init_fn=utils.seed_worker,
)

plot.show_image(
    loader=train_loader,
    nrows=1,
    ncols=1,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
```

## Forward Pass

As a sanity check of our model, we can perform a forward pass.

```python
forward_X, forward_y, model_summary = models.forward_pass(
    model=models.CustomNeuralNet()
)
```

## Augmentations

[Reference on RandomResizedCrop](https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/). So albumentation's has default parameters same as the article.


## Trainer

### Autocast

```python
if self.params.use_amp:
    self.optimizer.zero_grad()
    with torch.cuda.amp.autocast(
        enabled=True, dtype=torch.float16, cache_enabled=True
    ):
        logits = self.model(inputs)  # Forward pass logits
        curr_batch_train_loss = self.train_criterion(
            targets,
            logits,
            batch_size,
            criterion_params=CRITERION_PARAMS,
        )
    self.scaler.scale(curr_batch_train_loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
else:
    logits = self.model(inputs)  # Forward pass logits
    self.optimizer.zero_grad()  # reset gradients
    curr_batch_train_loss = self.train_criterion(
        targets,
        logits,
        batch_size,
        criterion_params=CRITERION_PARAMS,
    )
    curr_batch_train_loss.backward()  # Backward pass
    self.optimizer.step()  # Update weights using the optimizer
```
can be changed to 

```python
with torch.cuda.amp.autocast(
    enabled=self.params.use_amp,
    dtype=torch.float16,
    cache_enabled=True,
):
    logits = self.model(inputs)  # Forward pass logits
    curr_batch_train_loss = self.train_criterion(
        targets,
        logits,
        batch_size,
        criterion_params=CRITERION_PARAMS,
    )
self.optimizer.zero_grad()  # reset gradients

if self.scaler is not None:
    self.scaler.scale(curr_batch_train_loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
else:
    curr_batch_train_loss.backward()  # Backward pass
    self.optimizer.step()  # Update weights using the optimizer
```

This is in line with [torch's example](https://github.com/pytorch/vision/blob/main/references/classification/train.py). We reduced some overhead because we not longer need to check `if-else` for whether we use autocast or not. In our `config`, we already have a boolean flag `use_amp`, we just need to pass it to the `enabled` argument of `autocast` to indicate whether we are training with autocast or not.

## Miscellaneous

### Static Methods

In `trainer.py`, we used quite a few static methods. According to [geeksforgeeks](https://www.geeksforgeeks.org/class-method-vs-static-method-python/), static methods are methods that are called without creating an instance of the class, and are often used as utility functions. One immediate convenience is the method `get_sigmoid_softmax()` which returns either `nn.Sigmoid()` or `nn.Softmax()` depending on the loss function. I then also need to use it again in `inference.py`. I can just do the following:

```python
from src import trainer
print(trainer.Trainer.get_sigmoid_softmax()) -> nn.Sigmoid()
```

Notice that I can use the `get_sigmoid_softmax()` method in `inference.py` without creating an instance of the class.


## Coding Pitfalls

### Mutable Default Arguments

https://docs.python-guide.org/writing/gotchas/

### Dataclasses Call Functions Below

```python
def get_model_artifacts_path(self) -> Path:
    """Returns the model artifacts path.

    Returns:
        Path(model_artifacts_path) (Path): Model artifacts path.
    """
    # model_artifacts_path stores model weights, oof, etc. Note that now the model save path has wandb_run's group id appended for me to easily recover which run corresponds to which model.
    # create model directory if not exist and model_directory with run_id to identify easily.

    model_artifacts_path: Path = Path(
        self.weight_path,
        f"{ModelParams().model_name}_{WandbParams().group}",
    )
    Path.mkdir(model_artifacts_path, parents=True, exist_ok=True)
    # oof_csv: Path = Path(model_artifacts_path)
    return model_artifacts_path
```
In `FilePaths` we defined a method to call `WandbParams()` which is below this function. This is possible however if you do not define it in the method, it is not possible, maybe due to late closure binding?