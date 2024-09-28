"""A package for tuning generative models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, override

import dotenv
import google.generativeai as genai
import typer
import vertexai
from google.generativeai.types.model_types import TuningExampleDict
from loguru import logger
from openai import OpenAI
from openai.types.fine_tuning.job_create_params import Hyperparameters
from typing_extensions import Annotated
from vertexai.tuning import sft

import babbler
from babbler.resources import Provider, JsonModel


class TunePipeline(ABC):
    """A class for running tasks."""

    @abstractmethod
    def run(self) -> None:
        """Runs the pipeline."""
        raise NotImplementedError


class TuneConfig(JsonModel):
    """Configuration options for tuning models.

    Not all options are supported across all model platforms.
    """

    adapter_size: int | None = None
    """The adapter size to use for the tuning job, which influences the number of trainable parameters."""

    description: str | None = None
    """An optional description of the tuning model."""

    epochs: int | None = None
    """The number of epochs to train for."""

    batch_size: int | None = None
    """The training batch size."""

    name: str | None = None
    """A name to give the tuned model."""

    seed: int | None = None
    """The random seed for reproducibility."""

    temperature: float | None = None
    """The temperature for chat completion."""


class GoogleAITuner(TunePipeline):
    """Creates tuning jobs for Google AI models."""

    def __init__(
        self,
        model_name: str,
        train: str,
        tune_config: TuneConfig | None = None,
    ):
        """Create a new tuner.

        :param model_name: The name of the model to tune.
        :param train: Path to a training dataset.
        :param tune_config: Configures fine-tuning.
        """
        super().__init__()
        self.model_name = model_name
        self.train = train
        self.tune_config = tune_config or TuneConfig()

    @override
    def run(self) -> None:
        train_path = Path(self.train)
        if not train_path.exists():
            raise FileNotFoundError(f'Training dataset not found: {self.train}')
        training_data: list[TuningExampleDict] = []
        for blob in babbler.files.yield_jsonl(train_path):
            training_data.append(
                TuningExampleDict(
                    text_input=blob['text_input'],
                    output=blob['output'],
                )
            )
        genai.create_tuned_model(
            display_name=self.tune_config.name,
            source_model=self.model_name,
            epoch_count=self.tune_config.epochs,
            batch_size=self.tune_config.batch_size,
            training_data=training_data,
            temperature=self.tune_config.temperature,
            description=self.tune_config.description,
        )


class OpenAITuner(TunePipeline):
    """Creates tuning jobs for OpenAI models."""

    def __init__(
        self,
        model_name: str,
        train: str,
        validation: str | None = None,
        client: OpenAI | None = None,
        tune_config: TuneConfig | None = None,
    ):
        """Create a new tuner.

        :param model_name: The name of the model to tune.
        :param train: An OpenAI file ID of a training dataset.
        :param validation: An OpenAI file ID of a validation dataset.
        :param client: The client to use for API calls. A default client is created if not
            specified.
        :param tune_config: Configures fine-tuning.
        """
        super().__init__()
        self.model_name = model_name
        self.train = train
        self.validation = validation
        self.client = client or OpenAI()
        self.tune_config = tune_config or TuneConfig()

    @override
    def run(self) -> None:
        hyperparameters = Hyperparameters(
            n_epochs=self.tune_config.epochs or 'auto',
            batch_size=self.tune_config.batch_size or 'auto',
        )
        job = self.client.fine_tuning.jobs.create(
            training_file=self.train,
            validation_file=self.validation,
            model=self.model_name,
            seed=self.tune_config.seed,
            suffix=self.tune_config.name,
            hyperparameters=hyperparameters,
        )
        logger.info(f'Created tuning job: {job}')


class VertexAITuner(TunePipeline):
    """Creates tuning jobs for Vertex AI models."""

    def __init__(
        self,
        model_name: str,
        train: str,
        validation: str | None = None,
        tune_config: TuneConfig | None = None,
    ):
        """Create a new tuner.

        :param model_name: The name of the model to tune.
        :param train: A Cloud Storage path to a training dataset.
        :param validation: A Cloud Storage path to a validation dataset.
        :param tune_config: Configures fine-tuning.
        """
        super().__init__()
        self.model_name = model_name
        self.train = train
        self.validation = validation
        self.tune_config = tune_config or TuneConfig()

    @override
    def run(self) -> None:
        job = sft.train(
            source_model=self.model_name,
            train_dataset=self.train,
            validation_dataset=self.validation,
            epochs=self.tune_config.epochs,
            adapter_size=self.tune_config.adapter_size,
            tuned_model_display_name=self.tune_config.name,
        )
        logger.info(f'Created tuning job: {job}')


def fine_tune(
    train_uri: Annotated[
        str,
        typer.Option(
            default='--train',
            help='A URI to a training dataset.',
        ),
    ],
    provider: Annotated[Provider, typer.Option(help='The model provider.')],
    model: Annotated[
        str,
        typer.Option(
            help='The base model to tune.',
        ),
    ],
    validation_uri: Annotated[
        Optional[str],
        typer.Option(
            default='--validation',
            help='An optional URI to a validation dataset.',
        ),
    ] = None,
    tune_config_path: Annotated[
        Optional[Path],
        typer.Option(
            default='--tune-config',
            help='An optional path to a tuning configuration.',
            dir_okay=False,
        ),
    ] = None,
    env: Annotated[
        Optional[Path],
        typer.Option(
            help='An optional path to a .env file.',
            dir_okay=False,
        ),
    ] = None,
    vertexai_project: Annotated[
        Optional[str],
        typer.Option(
            help='The Google Cloud project to use with Vertex AI. E.g.: "123456789000"',
        ),
    ] = None,
    vertexai_location: Annotated[
        Optional[str],
        typer.Option(
            help='The Google Cloud region or location to use with Vertex AI. E.g.: "europe-west3"',
        ),
    ] = None,
):
    """Tune a generative model."""
    dotenv.load_dotenv(dotenv_path=env)
    tune_config: TuneConfig | None = None
    if tune_config_path and tune_config_path.exists():
        tune_config = TuneConfig.from_json(tune_config_path)
    pipeline: TunePipeline
    if provider == Provider.google_ai:
        # validation data is not supported by Google AI
        pipeline = GoogleAITuner(
            train=train_uri,
            model_name=model,
            tune_config=tune_config,
        )
    elif provider == Provider.openai:
        pipeline = OpenAITuner(
            train=train_uri,
            validation=validation_uri,
            model_name=model,
            tune_config=tune_config,
        )
    elif provider == Provider.vertexai:
        vertexai.init(project=vertexai_project, location=vertexai_location)
        pipeline = VertexAITuner(
            train=train_uri,
            validation=validation_uri,
            model_name=model,
            tune_config=tune_config,
        )
    else:
        raise ValueError(f'Unsupported provider: {provider}')
    pipeline.run()
