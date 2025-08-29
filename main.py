import yaml
from pathlib import Path
from RLHF.main import train as rlhf_train


def main():
    # Use root config.yaml
    config_path = Path("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    rlhf_train(config)


if __name__ == "__main__":
    main()


