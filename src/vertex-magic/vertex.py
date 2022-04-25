from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.core.magic_arguments import (argument, magic_arguments,
                                        parse_argstring)
import google.auth
from datetime import datetime
from google.cloud import aiplatform as aip
import os


class Context(object):
    """Storage for objects to be used throughout an IPython notebook session.
    A Context object is initialized when the ``magics`` module is imported.
    """

    def __init__(self):
        self._credentials = None
        self._project = None
        
    @property
    def credentials(self):
        """google.auth.credentials.Credentials: Credentials to use for queries
        performed through IPython magics.
        Note:
            These credentials do not need to be explicitly defined if you are
            using Application Default Credentials. If you are not using
            Application Default Credentials, manually construct a
            :class:`google.auth.credentials.Credentials` object and set it as
            the context credentials as demonstrated in the example below. See
            `auth docs`_ for more information on obtaining credentials.
        Example:
            Manually setting the context credentials:
            >>> from google.cloud.bigquery import magics
            >>> from google.oauth2 import service_account
            >>> credentials = (service_account
            ...     .Credentials.from_service_account_file(
            ...         '/path/to/key.json'))
            >>> magics.context.credentials = credentials
        .. _auth docs: http://google-auth.readthedocs.io
            /en/latest/user-guide.html#obtaining-credentials
        """
        if self._credentials is None:
            self._credentials, _ = google.auth.default()
        return self._credentials

    @credentials.setter
    def credentials(self, value):
        self._credentials = value

    @property
    def project(self):
        """str: Default project to use for queries performed through IPython
        magics.
        Note:
            The project does not need to be explicitly defined if you have an
            environment default project set. If you do not have a default
            project set in your environment, manually assign the project as
            demonstrated in the example below.
        Example:
            Manually setting the context project:
            >>> from google.cloud.bigquery import magics
            >>> magics.context.project = 'my-project'
        """
        if self._project is None:
            _, self._project = google.auth.default()
        return self._project

    @project.setter
    def project(self, value):
        self._project = value

        
context = Context()        
    
def get_train_image(gpus_nb:int=0):
    if gpus_nb > 0:
        TRAIN_GPU, TRAIN_NGPU = (
            aip.gapic.AcceleratorType.NVIDIA_TESLA_K80,
            gpus_nb,
        )
    else:
        TRAIN_GPU, TRAIN_NGPU = (None, None)

    TF = "2-1"
    if TF[0] == "2":
        if TRAIN_GPU:
            TRAIN_VERSION = "tf-gpu.{}".format(TF)
        else:
            TRAIN_VERSION = "tf-cpu.{}".format(TF)

    TRAIN_IMAGE = "{}-docker.pkg.dev/vertex-ai/training/{}:latest".format(
        REGION.split("-")[0], TRAIN_VERSION
    )
    DEPLOY_IMAGE = "{}-docker.pkg.dev/vertex-ai/prediction/{}:latest".format(
        REGION.split("-")[0], DEPLOY_VERSION
    )

    print("Training:", TRAIN_IMAGE, TRAIN_GPU, TRAIN_NGPU)
    print("Deployment:", DEPLOY_IMAGE, DEPLOY_GPU, DEPLOY_NGPU)
    
@magics_class
class Vertex(Magics):
    
    @magic_arguments()
    @argument(
        "--region",
        type=str,
        default=None,
        help=("Location to run training in."),
    )
    @cell_magic
    def vertex(self, line, cell):


        args = parse_argstring(self.vertex, line)

        TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
        training_python_file = f"cell_{TIMESTAMP}.py"
        with open(training_python_file, "w") as text_file:
            text_file.write(cell)


        REGION = "europe-west4"

        aip.init(project=context.project, credentials=context.credentials, staging_bucket=f"gs://{context.project}", location=args.region)

        DISPLAY_NAME = "magic_" + TIMESTAMP
        REQUIREMENTS = []
        TRAIN_IMAGE = "europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest"

        job = aip.CustomTrainingJob(
            display_name=DISPLAY_NAME,
            script_path=training_python_file,
            requirements=REQUIREMENTS,
            container_uri=TRAIN_IMAGE,
        )

        CMDARGS = []
        TRAIN_COMPUTE = "n1-standard-4"

        job.run(args=CMDARGS, replica_count=1, machine_type=TRAIN_COMPUTE, sync=True)
        os.remove(training_python_file)
