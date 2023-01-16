# ZenBytes

ZenBytes is a series of short practical MLOps lessons through [ZenML](https://github.com/zenml-io/zenml) and its various integrations. It is intended for people looking to learn about MLOps generally, and also for ML practitioners who want to get started with ZenML.

## :triangular_flag_on_post: ZenML 0.20.0 chapters coming soon!

The release of ZenML 0.20.0 marks a big breaking change in ZenML history, and requires an equally big update to this repository. The ZenML team is working on this renovation as we speak, and will bring you a brand-new ZenBytes with the latest ZenML soon!

## :bulb: What you will learn
- Define an MLOps stack tailored to your project requirements.
- Build transparent and reproducible data-centric ML pipelines with automated artifact versioning, tracking, caching, and more.
- Deploy ML pipelines with tooling and infrastructure of your choice (e.g. as a serverless microservice in the cloud).
- Monitor and address production issues like training-serving skew and data drift.
- Use some of the most popular MLOps tools like ZenML, Kubeflow, MLflow, Weights & Biases, Evidently, Seldon, Feast, and many more.

In the end, you will be able to take any of your ML models from experimentation to a customized, fully fleshed-out production-grade MLOps setup in a matter of minutes!

<div align="center">
<img src="_assets/sam.png" alt="Sam"/>
</div>

## :teacher: Syllabus

The series is structured into four chapters with several lessons each. Click on any of the links below to open the respective lesson directly in Colab.

| :dango: 1. ML Pipelines        | :recycle: 2. Training / Serving   | :file_folder: 3. Data Management  | :rocket: More Coming Soon! |
|------------------------|-------------------------|---------------------|------------------------|
| [1.1 ML Pipelines](https://colab.research.google.com/github/zenml-io/zenbytes/blob/main/1-1_Pipelines.ipynb)       | [2.1 Experiment Tracking](https://colab.research.google.com/github/zenml-io/zenbytes/blob/main/2-1_Experiment_Tracking.ipynb) | [3.1 Data Skew](https://colab.research.google.com/github/zenml-io/zenbytes/blob/main/3-1_Data_Skew.ipynb) |    |
| [1.2 Artifact Lifecycle](https://colab.research.google.com/github/zenml-io/zenbytes/blob/main/1-2_Artifact_Lineage.ipynb) | [2.2 Local Deployment](https://colab.research.google.com/github/zenml-io/zenbytes/blob/main/2-2_Local_Deployment.ipynb)    |                     |                        |
|                        | [2.3 Inference Pipelines](https://colab.research.google.com/github/zenml-io/zenbytes/blob/main/2-3_Inference_Pipelines.ipynb) |                     |                        |

<!--
### Syllabus Details:

- Chapter 1: ML Pipelines and Stacks
    - Lesson 1.1: ML Pipelines with ZenML
    - Lesson 1.2: Artifact Versioning, Tracking, and Caching
    - (Coming Soon: Choosing the Right Tools for your MLOps Stack)
- Chapter 2: Training, Deployment, and Serving
    - Lesson 2.1: Experiment Tracking with MLflow / W&B
    - Lesson 2.2: Local Deployment with MLflow
    - Lesson 2.3: Inference Pipelines
- Chapter 3: Data Management
    - Lesson 3.1: Data Drift Detection with Evidently / Whylabs
    - (Coming Soon: Data Validation with DeepChecks / Great Expectations)
     (Coming Soon: Feature Stores with Feast)
- Chapter 4: Advanced Deployment
    - Lesson 4.1: Scalable Cloud Deployment with Seldon & Kubeflow on AWS
    - (Coming Soon: Scalable Local Deployment with Seldon & Kubeflow)
- Chapter 5: Full Examples
    - (Lesson 5.1: Zero to Hero with ZenML - from Experimentation to Production-Grade MLOps)
    - (Lesson 5.2: More Examples - zenml example run and ZenML Projects)
- (unused)
    - (Materializers & skipping them)
    - (Lesson 3: Defining MLOps Stacks with ZenML? -> Profiles, Repos)
    - (Lesson 3.6: Running ZenML Steps on Specialized Hardware)
    - Lesson 3.5: Continuous Deployment based on Data Drift Triggers
- (missing functionality)
    - Model Registries
    - Explainability Tools
    - Model CI/CD
    - AutoML
-->

## :pray: About ZenML
ZenML is an extensible, open-source MLOps framework for creating production-ready ML pipelines. Built for data scientists, it has a simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that are catered towards ML workflows.

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
As you progress through the course, you will need to install additional
packages for the various other MLOps tools we will use.
You will find corresponding instructions in the respective notebooks,
but we recommend you install all integrations ahead of time with the
following command:

```bash
zenml integration install sklearn wandb evidently mlflow -y
```

## :rocket: Getting Started

If you haven't done so already, clone ZenBytes to your local machine. 
Then, use Jupyter Notebook to go through the course lesson-by-lesson, starting 
with `1-1_Pipelines.ipynb`:

```bash
git clone https://github.com/zenml-io/zenbytes
cd zenbytes
jupyter notebook
```

## :question: FAQ


#### 1. ZenML cannot find a component even though I have it in my stack
Updating or switching your ZenML stack is sometimes not immediately 
loaded in Jupyter notebooks.

**Solution:** First, make sure you really have the correct component installed
and registered in your currently active stack with `zenml stack describe`.
If the component is indeed there, **restart the kernel** of your Jupyter notebook,
which will also reload the stack.
