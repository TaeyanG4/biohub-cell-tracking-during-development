from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    result = api.competition_submit_code(
        file_name="submission.csv",
        message="clean public-0.947 settings anchor",
        competition="biohub-cell-tracking-during-development",
        kernel="taeyangg4/biohub-dctta-lite-public0947-anchor",
        kernel_version=1,
        quiet=False,
    )
    print(result)


if __name__ == "__main__":
    main()
