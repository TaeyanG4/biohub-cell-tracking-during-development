param([Parameter(Mandatory=$true)][ValidateSet('c067','c068')][string]$Candidate)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskGroup=if($Candidate -eq 'c067'){'44b6'}else{'6bba'}
$taskRelative="experiments/candidates/${Candidate}_single_source_${taskGroup}/local"
$taskOut=Join-Path $taskRoot $taskRelative
if((Test-Path -LiteralPath (Join-Path $taskOut 'launch.json')) -or (Test-Path -LiteralPath (Join-Path $taskOut 'status.json'))){throw 'Existing local phase; inspect before any recovery'}
$taskPlan=Get-Content -LiteralPath (Join-Path $taskOut 'plan.json') -Raw|ConvertFrom-Json
if([DateTimeOffset]::Now.AddMinutes([int]$taskPlan.estimated_minutes+60) -ge [DateTimeOffset]::Parse('2026-09-30T00:00:00+09:00')){throw 'Insufficient time for local review plus clean T4/submission'}
$taskObserve=Join-Path $taskRoot 'state/background_notifications'
$taskPython='C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe'
$taskPrefix=$Candidate+'_local_review'
$taskArgs=@('-X','utf8','-u','src/c067_collect.py','run','--candidate',$Candidate)
$taskProcess=Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_owner.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_owner.stderr.log')) -PassThru
$taskStart=[DateTimeOffset]::Now
$taskEta=$taskStart.AddMinutes([int]$taskPlan.estimated_minutes)
$taskLaunch=[ordered]@{candidate=$Candidate;phase='local_actual97_portable';pid=$taskProcess.Id;process_start_time=$taskProcess.StartTime.ToString('o');started_at=$taskStart.ToString('o');estimated_completion=$taskEta.ToString('o');first_check=$taskEta.AddMinutes(-10).ToString('o');check_interval_minutes=20;estimated_minutes=$taskPlan.estimated_minutes;total_jobs=$taskPlan.total_jobs;source='src/c067_collect.py';source_sha256=(Get-FileHash (Join-Path $taskRoot 'src/c067_collect.py') -Algorithm SHA256).Hash.ToLower();no_gpu_inference=$true;no_kaggle_writes=$true}
$taskLaunch|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $taskOut 'launch.json') -Encoding UTF8
$taskWatchArgs=@('-NoProfile','-ExecutionPolicy','Bypass','-File','tools/notify_background_completion.ps1','-Mode','Watch','-Label',($Candidate.ToUpper()+'-local-review'),'-StatusPath',($taskRelative+'/status.json'),'-LaunchPath',($taskRelative+'/launch.json'),'-ReceiptPath',('state/background_notifications/'+$taskPrefix+'_completion.json'))
$taskWatcher=Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList $taskWatchArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_watcher.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_watcher.stderr.log')) -PassThru
@{watcher_pid=$taskWatcher.Id;owner_pid=$taskProcess.Id;started_at=$taskStart.ToString('o');logs_outside_candidate=$true;independent_of_heartbeat=$true}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $taskObserve ($taskPrefix+'_watcher_launch.json')) -Encoding UTF8
$taskMonitorPath=Join-Path $taskRoot 'state/biohub_t4_monitor.json'
$taskMonitor=Get-Content -LiteralPath $taskMonitorPath -Raw|ConvertFrom-Json
$taskMonitor.local_queues=@($taskMonitor.local_queues|Where-Object {$_.candidate -ne $Candidate})+@([pscustomobject]@{candidate=$Candidate;phase='local_actual97_portable';pid=$taskProcess.Id;status_path=$taskRelative+'/status.json';launch_path=$taskRelative+'/launch.json';estimated_completion=$taskEta.ToString('o')})
$taskMonitor.updated=$taskStart.ToString('o')
$taskMonitor|ConvertTo-Json -Depth 20|Set-Content -LiteralPath $taskMonitorPath -Encoding UTF8
$taskLaunch|ConvertTo-Json -Depth 8
