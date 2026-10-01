import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "biohub-cell-tracking-during-development"
KERNEL = "taeyangg4/biohub-c004-adaptive-lineage"
KERNEL_VERSION = 1
MESSAGE = "C004 adaptive lineage enveloping (sister 16.0, exist 12.0, diverge 0.5, sym 0.85, margin 0.001)"


def main() -> None:
    api = KaggleApi()
    api.authenticate()

    try:
        status_obj = api.kernels_status(KERNEL)
        status = status_obj.status
        print(f"Kernel status: {status}")
    except Exception as exc:
        print(f"Kernel status check: {exc}")

    try:
        result = api.competition_submit_code(
            file_name="submission.csv",
            message=MESSAGE,
            competition=COMPETITION,
            kernel=KERNEL,
            kernel_version=KERNEL_VERSION,
            quiet=False,
        )
        print("Submit response:", result)
    except Exception as exc:
        print(f"Submit attempt error ({type(exc).__name__}): {exc}")


if __name__ == "__main__":
    main()
