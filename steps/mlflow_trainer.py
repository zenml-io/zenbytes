from steps.trainer import svc_trainer_mlflow, tree_trainer_with_mlflow
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow


# These are the same steps, but with mlflow enabled
svc_trainer_mlflow = enable_mlflow(svc_trainer_mlflow)
tree_trainer_with_mlflow = enable_mlflow(tree_trainer_with_mlflow)
