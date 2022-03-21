# ZenBytes

![ZenML Logo](_assets/Logo/zenml.svg)


<div align="center">
Join our <a href="https://zenml.io/slack-invite" target="_blank">
    <img width="25" src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/306_Slack-512.png" alt="Slack"/>
<b>Slack Community</b> </a> and become part of the ZenML family
</div>
<div align="center"> Give us a 
    <img width="25" src="https://cdn.iconscout.com/icon/free/png-256/github-153-675523.png" alt="Slack"/>
<b>GitHub star</b> to show your love
</div>

<br>

A simple guide to MLOps through [ZenML](https://github.com/zenml-io/zenml) and its various integrations. This repository is still a WIP. Please start with Chapter 0 in 
`Chapter 000 - Basics of ZenML`


![ZenML Logo](_assets/evidently+mlflow+discord+kubeflow.png)


ZenML is an extensible, open-source MLOps framework to create production-ready machine learning pipelines. Built for data scientists, it has a simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that are catered towards ML workflows.

Check out the [ZenML repository](https://github.com/zenml-io/zenml) and [Docs](https://docs.zenml.io) for more details.

## :computer: System Requirements

In order to run this entire demo you need to have some packages installed on your machine. Note you only need these for some parts, and you might get away 
with only Python and `pip install requirements.txt` for some parts of the codebase, but we recommend installing all these:

Currently, this will only run on UNIX systems.

| package  | MacOS installation                                                               | Linux installation                                                                 |
|----------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| docker   | [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)           | [Docker Engine for Linux ](https://docs.docker.com/engine/install/ubuntu/)         |
| kubectl  | [kubectl for mac](https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/) | [kubectl for linux](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) |
| k3d      | [Brew Installation of k3d](https://formulae.brew.sh/formula/k3d)                 | [k3d installation linux](https://k3d.io/v5.2.2/)                                   |

You might also need to install [Anaconda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) to get the MLflow deployment to work.

## :snake: Python Requirements

Once you've got the system requirements figured out, let's jump into the Python packages you need. 
Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/
cd "Chapter 000 - Basics of ZenML"
pip install -r requirements.txt
```

If you are running the `run.py` script, you will also need to install some integrations using zenml:

```bash
zenml integration install evidently -f
zenml integration install mlflow -f
zenml integration install kubeflow -f
```

## :notebook: Diving into the code

We're ready to go now. You have two options:

### Notebook

You can go through the notebook step-by-step guide:

```python
jupyter notebook
```

### Script

You can also directly run the code, using the `run.py` script.

```python
zenml init
python run.py  # Runs pipeline locally
```

## :checkered_flag: Cleaning up when you're done

Once you are done running all notebooks you might want to stop all running processes. For this, run the following command.
(This will tear down your `k3d` cluster and the local docker registry.)


```shell
zenml stack set local_kubeflow_stack
zenml stack down -f
```

## :question: FAQ

1. __MacOS__ When starting the container registry for Kubeflow, I get an error about port 5000 not being available.
`OSError: [Errno 48] Address already in use`

Solution: In order for Kubeflow to run, the docker container registry currently needs to be at port 5000. MacOS, however, uses 
port 5000 for the Airplay receiver. Here is a guide on how to fix this [Freeing up port 5000](https://12ft.io/proxy?q=https%3A%2F%2Fanandtripathi5.medium.com%2Fport-5000-already-in-use-macos-monterey-issue-d86b02edd36c).
