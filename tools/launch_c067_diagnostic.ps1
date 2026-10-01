param([ValidateSet('c067','c068')][string]$Candidate='c067')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskGroup=if($Candidate -eq 'c067'){'44b6'}else{'6bba'}
$taskRelative="experiments/candidates/${Candidate}_single_source_${taskGroup}/diagnostic"
$taskOut=Join-Path $taskRoot $taskRelative
if((Test-Path -LiteralPath (Join-Path $taskOut 'launch.json')) -or (Test-Path -LiteralPath (Join-Path $taskOut 'status.json'))){throw 'Existing diagnostic; no blind relaunch'}
$taskObserve=Join-Path $taskRoot 'state/background_notifications'
$taskPrefix=$Candidate+'_diagnostic'
$taskProcess=Start-Process -FilePath 'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe' -ArgumentList @('-X','utf8','-u','state/c067_diagnostic.py','run','--candidate',$Candidate) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_owner.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_owner.stderr.log')) -PassThru
$taskStart=[DateTimeOffset]::Now
$taskEta=$taskStart.AddMinutes(20)
$taskLaunch=[ordered]@{candidate=$Candidate;phase='saved_graph_diagnostic_only';pid=$taskProcess.Id;process_start_time=$taskProcess.StartTime.ToString('o');started_at=$taskStart.ToString('o');estimated_completion=$taskEta.ToString('o');first_check=$taskEta.AddMinutes(-10).ToString('o');check_interval_minutes=20;estimated_minutes=20;estimate_basis='53k input rehash about5min plus97 actual organizer calls about10min plus5min margin';source='state/c067_diagnostic.py';source_sha256=(Get-FileHash (Join-Path $taskRoot 'state/c067_diagnostic.py') -Algorithm SHA256).Hash.ToLower();no_gpu_inference=$true;no_kaggle_writes=$true;not_submission_eligible=$true}
$taskLaunch|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $taskOut 'launch.json') -Encoding UTF8
$taskWatchArgs=@('-NoProfile','-ExecutionPolicy','Bypass','-File','tools/notify_background_completion.ps1','-Mode','Watch','-Label',($Candidate.ToUpper()+'-diagnostic'),'-StatusPath',($taskRelative+'/status.json'),'-LaunchPath',($taskRelative+'/launch.json'),'-ReceiptPath',('state/background_notifications/'+$taskPrefix+'_completion.json'))
$taskWatcher=Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList $taskWatchArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_watcher.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_watcher.stderr.log')) -PassThru
@{watcher_pid=$taskWatcher.Id;owner_pid=$taskProcess.Id;started_at=$taskStart.ToString('o');logs_outside_candidate=$true}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $taskObserve ($taskPrefix+'_watcher_launch.json')) -Encoding UTF8
$taskMonitorPath=Join-Path $taskRoot 'state/biohub_t4_monitor.json'
$taskMonitor=Get-Content -LiteralPath $taskMonitorPath -Raw|ConvertFrom-Json
$taskMonitor.local_queues=@($taskMonitor.local_queues|Where-Object {$_.candidate -ne $Candidate})+@([pscustomobject]@{candidate=$Candidate;phase='saved_graph_diagnostic_only';pid=$taskProcess.Id;status_path=$taskRelative+'/status.json';launch_path=$taskRelative+'/launch.json';estimated_completion=$taskEta.ToString('o')})
$taskMonitor.updated=$taskStart.ToString('o')
$taskMonitor|ConvertTo-Json -Depth 20|Set-Content -LiteralPath $taskMonitorPath -Encoding UTF8
$taskLaunch|ConvertTo-Json -Depth 8
