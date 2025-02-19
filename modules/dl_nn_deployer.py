import subprocess
from . import utils  # Import utils

class DLNNDeployer:
    def __init__(self, config):
        self.config = config

    def deploy_bot(self, bot_path):
        cmd = f"python3 {bot_path}"
        subprocess.run(cmd, shell=True)
        utils.log("DL/NN Bot deployed successfully!")
