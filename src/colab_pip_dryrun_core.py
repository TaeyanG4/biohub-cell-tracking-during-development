import subprocess, sys
cmd=[sys.executable,'-m','pip','install','--dry-run','tracksdata==0.1.0rc10','spatial-graph==0.1.1','zarr==3.2.1','gurobipy==12.0.3','pydantic>=2.12.5','pyyaml>=6.0.3','rich>=14.3.1']
print('running',' '.join(cmd),flush=True)
p=subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(p.stdout)
print('returncode',p.returncode)
