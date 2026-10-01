import sys
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "biohub-cell-tracking-during-development"
KERNEL = "taeyangg4/biohub-c003-medal-frontier"
KERNEL_VERSION = 1
MESSAGE = "C003 medal frontier (diverge 1.0, sym 0.8, tight 5.5, expanded PP)"


def main() -> None:
    api = KaggleApi()
    api.authenticate()

    status_obj = api.kernels_status(KERNEL)
    status = status_obj.status
    print(f"Kernel status: {status}")

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
