from __future__ import annotations

from kaggle.api.kaggle_api_extended import KaggleApi


COMPETITION = "biohub-cell-tracking-during-development"
TARGET_PREFIXES = [
    "train/44b6_0113de3b.geff",
    "train/44b6_0b24845f.geff",
    "train/6bba_05b6850b.geff",
    "train/6bba_05db0fb1.geff",
]


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    token = None
    page = 0
    found: list[str] = []

    while True:
        response = api.competition_list_files(
            COMPETITION,
            page_token=token,
            page_size=200,
        )
        page += 1
        for item in response.files:
            if any(item.name.startswith(prefix) for prefix in TARGET_PREFIXES):
                found.append(item.name)
        if page % 20 == 0:
            print(f"page={page} found={len(found)}", flush=True)
        token = getattr(response, "next_page_token", None)
        if not token or page >= 250:
            break

    print(f"pages={page} found={len(found)}")
    for name in found:
        print(name)


if __name__ == "__main__":
    main()
