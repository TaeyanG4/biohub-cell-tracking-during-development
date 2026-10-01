import subprocess

cmd = [
    "kaggle",
    "kernels",
    "push",
    "-p",
    "experiments/exp_dctta_hoct_det0965_nolinefit_mode1",
    "--accelerator",
    "NvidiaTeslaT4",
]
raise SystemExit(subprocess.call(cmd))
