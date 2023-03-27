import os
import subprocess


def up_tensorboard_port(model_name: str) -> None:
    """
    Launch a TensorBoard instance for the given model.
    Parameters
    ----------
    model_name : str
        The name of the model whose TensorBoard instance should be launched.
    """
    cmd = ["tensorboard", "--logdir", f"./reports/{model_name}"]
    cmd += ["--port", "9999"]
    subprocess.Popen(cmd, shell=False)


def down_tensorboard_port() -> None:
    """
    Kill any existing tensorboard process running on the system.
    """
    cmd = "kill $(ps -e | grep 'tensorboard' | awk '{print $1}')"
    os.system(cmd)
