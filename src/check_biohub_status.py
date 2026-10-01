import sys
import io
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

COMPETITION = "biohub-cell-tracking-during-development"

KERNELS = [
    "taeyangg4/biohub-c003-medal-frontier",
    "taeyangg4/biohub-c004-adaptive-lineage",
    "taeyangg4/biohub-dctta-lite-public0947-anchor",
    "taeyangg4/biohub-dctta-dualhoct-det0965-nolinefit",
    "taeyangg4/biohub-dctta-hoct-det0965-nolinefit",
    "taeyangg4/biohub-dctta-hoct-det0965",
    "taeyangg4/biohub-edge-tta-det-0-96-experiment",
]


def main() -> None:
    api = KaggleApi()
    api.authenticate()

    print("KERNELS")
    for name in KERNELS:
        try:
            status = api.kernels_status(name)
            print(f"{name}: {status}")
        except Exception as exc:
            msg = str(exc)
            if "Cannot access kernel" in msg or "403" in msg:
                print(f"{name}: NOT_PUSHED_YET (local only)")
            else:
                print(f"{name}: {type(exc).__name__} ({msg[:100]})")

    print("\nRECENT SUBMISSIONS")
    try:
        subs = api.competition_submissions(COMPETITION)
        for item in subs[:12]:
            print(f"{item.ref} | {item.status} | {item.public_score} | {item.description}")
    except Exception as exc:
        print("Submissions error:", exc)


if __name__ == "__main__":
    main()
