# ZenBytes

ZenBytes is a series of short practical MLOps lessons through [ZenML](https://github.com/zenml-io/zenml) and its various integrations. It is intended for people looking to learn about MLOps generally, and also for ML practitioners who want to get started with ZenML.

## :bulb: What you will learn
- Define an MLOps stack tailored to your project requirements.
- Build transparent and reproducible data-centric ML pipelines with automated artifact versioning, tracking, caching, and more.
- Deploy ML pipelines with tooling and infrastructure of your choice (e.g. as serverless microservice in the cloud).
- Monitor and address production issues like performance drift, data drift, and concept drift.
- Use some of the most popular MLOps tools like ZenML, Kubeflow, MLflow, Weights & Biases, Evidently, Seldon, Feast, and many more.

In the end, you will be able to take any of your ML models from experimentation to a customized, fully fleshed out production-grade MLOps setup in a matter of minutes!

<div align="center">
<img src="_assets/sam.png" alt="Sam"/>
</div>

## :teacher: Syllabus

- Chapter 1: ML Pipelines
    - Lesson 1.1: ML Pipelines
    - Lesson 1.2: Artifact Versioning, Tracking, and Caching
- Chapter 2: Transparency & Reproducibility
    - Lesson 2.1: Experiment Tracking with W&B / MLFlow
    - Lesson 2.2: Data Drift Detection with Evidently
    - Lesson 2.3: Automated Discord Alerts
    - (Lesson 2.4: Data Validation with DeepChecks / GreatExpectations)
    - (Lesson 2.5: Feature Stores with Feast?)
- Chapter 3: Deployment
    - Lesson 3.1: Local Deployment & Inference with MLFlow
    - Lesson 3.2: Continuous Deployment based on Data Drift Triggers
    - (Lesson 3.3: Model Serving with Seldon / BentoML?)
    - Lesson 3.4: Serverless Deployment with Seldon & Kubeflow
    - Lesson 3.5: Serverless Cloud Deployment with Seldon & Kubeflow on AWS (incl. Secret Managers)
- Chapter 4: Full Examples
    - (Lesson 4.1: Zero to Hero with ZenML - from Experimentation to Production-Grade MLOps)
    - (Lesson 4.2: More Examples - zenml example run and ZenFiles)

<!--
- (unused)
    - (Materializers & skipping them)
    - (Lesson 3: Defining MLOps Stacks with ZenML? -> Profiles, Repos)
    - (Lesson 3.6: Running ZenML Steps on Specialized Hardware)
- (missing functionality)
    - Model Registries
    - Explainability Tools
    - Model CI/CD
    - AutoML
-->

## :pray: About ZenML
ZenML is an extensible, open-source MLOps framework to create production-ready ML pipelines. Built for data scientists, it has a simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that are catered towards ML workflows. 

If you enjoy these courses and want to learn more:
- Give the <a href="https://github.com/zenml-io/zenml/stargazers" target="_blank">
    <img width="25" src="https://cdn.iconscout.com/icon/free/png-256/github-153-675523.png" alt="GitHub"/>
    <b>Main ZenML Repo</b>
</a> a <b>GitHub Star</b> :star: to show your love!
- Join our <a href="https://zenml.io/slack-invite" target="_blank">
    <img width="25" src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/306_Slack-512.png" alt="Slack"/>
    <b>Slack Community</b> 
</a> and become part of the ZenML family!

## :computer: Setup
### System Requirements

- Linux or MacOS
- Python 3.7 or 3.8
- Jupyter notebook and ZenML: `pip install zenml notebook`

### Integrations
As you progress through the course, you will need to install additional packages for the various other MLOps tools you are going to use.
You will find the corresponding commands in the respective notebooks. Or, you can install all of the integrations already with the following commands:

```bash
zenml integration install sklearn -f
zenml integration install dash -f
zenml integration install evidently -f
zenml integration install mlflow -f
zenml integration install kubeflow -f
zenml integration install seldon -f
```

### Additional Requirements
For some of the advanced lessons you also need to have the following additional packages installed on your machine:

| package  | MacOS installation                                                               | Linux installation                                                                 |
|----------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| docker   | [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)           | [Docker Engine for Linux ](https://docs.docker.com/engine/install/ubuntu/)         |
| kubectl  | [kubectl for mac](https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/) | [kubectl for linux](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) |
| k3d      | [Brew Installation of k3d](https://formulae.brew.sh/formula/k3d)                 | [k3d installation linux](https://k3d.io/v5.2.2/)                                   |

## :rocket: Getting Started

If you haven't done so already, clone ZenBytes to your local machine. Then, simply use Jupyter Notebook to go through the course lesson-by-lesson, starting with `1-1_Pipelines.ipynb`:


```bash
git clone https://github.com/zenml-io/zenbytes
cd zenbytes
jupyter notebook
```

## :question: FAQ

#### 1. __MacOS__ When starting the container registry for Kubeflow, I get an error about port 5000 not being available.
`OSError: [Errno 48] Address already in use`

Solution: In order for Kubeflow to run, the docker container registry currently needs to be at port 5000. MacOS, however, uses 
port 5000 for the Airplay receiver. Here is a guide on how to fix this [Freeing up port 5000](https://12ft.io/proxy?q=https%3A%2F%2Fanandtripathi5.medium.com%2Fport-5000-already-in-use-macos-monterey-issue-d86b02edd36c).
