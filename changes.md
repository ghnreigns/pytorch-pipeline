the main takeaway is that we should unify the configurations. I have many dataclasses:

```python
class Pipeline:
    
    model_params: ModelParams
    global_train_params: GlobalTrainParams
    ...
    
    def __init__(self, model_params, global_train_params, ...):
        self.model_params = model_params
        ...
    
    def train_one_fold():
        """Train the model on the given fold."""
    
    def run():
        """Run Typer application for pipeline."""
        app = typer.App()

pipeline = Pipeline(my_config...)

pipeline.run()     
```

So, we should unify the configurations so that our `Pipline` class can be called everywhere like:

```python
FILES = global_params.FilePaths()
FOLDS = global_params.MakeFolds()
LOADER_PARAMS = global_params.DataLoaderParams()
TRAIN_PARAMS = global_params.GlobalTrainParams()


def return_filepath(
    image_id: str,
    folder: Path = FILES.train_images,
    extension: str = FOLDS.image_extension,
) -> str:
    """Add a new column image_path to the train and test csv.
    We can call the images easily in __getitem__ in Dataset.

    We need to be careful if the image_id has extension already.
    In this case, there is no need to add the extension.

    Args:
        image_id (str): The unique image id: 1000015157.jpg
        folder (Path, optional): The train folder. Defaults to FILES().train_images.

    Returns:
        image_path (str): The path to the image: "c:\\users\\reighns\\kaggle_projects\\cassava\\data\\train\\1000015157.jpg"
    """
    # TODO: Consider using Path instead os for consistency.
    image_path = os.path.join(folder, f"{image_id}{extension}")
    return image_path
```

to 

```python
def return_filepath(
    image_id: str,
    pipeline_config: Pipeline
) -> str:
    """Add a new column image_path to the train and test csv.
    We can call the images easily in __getitem__ in Dataset.

    We need to be careful if the image_id has extension already.
    In this case, there is no need to add the extension.

    Args:
        image_id (str): The unique image id: 1000015157.jpg
        folder (Path, optional): The train folder. Defaults to FILES().train_images.

    Returns:
        image_path (str): The path to the image: "c:\\users\\reighns\\kaggle_projects\\cassava\\data\\train\\1000015157.jpg"
    """
    # TODO: Consider using Path instead os for consistency.
    image_id = pipeline.FILES.train_images.image_id
    extension = pipeline.FOLDS.image_extension.image_extension
    image_path = os.path.join(folder, f"{image_id}{extension}")
    return image_path
```

See the distinction now, we do not keep call the global params in each script, but everything is now unified under pipeline object. However, do note I explicitly defined image_id and extension in the function so that we know what they represent, instead of just putting into os.path.join. May be a bit verbose but it makes the code more readable for me. or consider put in docstring..