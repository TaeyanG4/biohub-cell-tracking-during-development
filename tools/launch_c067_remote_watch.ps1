param([Parameter(Mandatory=$true)][ValidateSet('c067','c068')][string]$Candidate)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskGroup=if($Candidate -eq 'c067'){'44b6'}else{'6bba'}
$taskRelative="experiments/candidates/${Candidate}_single_source_${taskGroup}/remote_audit"
$taskOut=Join-Path $taskRoot $taskRelative
if(Test-Path -LiteralPath (Join-Path $taskOut 'launch.json')){throw 'Watcher launch already exists; inspect owned process before recovery'}
$taskAction=Get-Content -LiteralPath (Join-Path $taskOut 'push_action.json') -Raw|ConvertFrom-Json
if($taskAction.status -ne 'pushed'){throw 'Research push not confirmed'}
$taskPlan=Get-Content -LiteralPath (Join-Path $taskRoot "experiments/candidates/${Candidate}_single_source_${taskGroup}/plan.json") -Raw|ConvertFrom-Json
$taskObserve=Join-Path $taskRoot 'state/background_notifications'
$taskPython='C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe'
$taskPrefix=$Candidate+'_remote_audit'
$taskArgs=@('-X','utf8','-u','src/c067_remote_research.py','watch','--candidate',$Candidate)
$taskProcess=Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_owner.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_owner.stderr.log')) -PassThru
$taskStart=[DateTimeOffset]::Now
$taskRemoteStart=[DateTimeOffset]::Parse($taskAction.ended)
$taskEta=$taskRemoteStart.AddMinutes([int]$taskPlan.estimated_remote_minutes)
$taskLaunch=[ordered]@{candidate=$Candidate;phase='private_T4_train_audit';pid=$taskProcess.Id;process_start_time=$taskProcess.StartTime.ToString('o');started_at=$taskStart.ToString('o');remote_pushed_at=$taskRemoteStart.ToString('o');remote_ref=$taskAction.ref;version=$taskAction.version;estimated_completion=$taskEta.ToString('o');first_check=$taskEta.AddMinutes(-10).ToString('o');check_interval_minutes=20;estimated_minutes=$taskPlan.estimated_remote_minutes;source='src/c067_remote_research.py';source_sha256=(Get-FileHash (Join-Path $taskRoot 'src/c067_remote_research.py') -Algorithm SHA256).Hash.ToLower();no_competition_submission=$true;research_only_never_submit=$true}
$taskLaunch|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $taskOut 'launch.json') -Encoding UTF8
$taskWatchArgs=@('-NoProfile','-ExecutionPolicy','Bypass','-File','tools/notify_background_completion.ps1','-Mode','Watch','-Label',($Candidate.ToUpper()+'-T4-research'),'-StatusPath',($taskRelative+'/status.json'),'-LaunchPath',($taskRelative+'/launch.json'),'-ReceiptPath',('state/background_notifications/'+$taskPrefix+'_completion.json'))
$taskWatcher=Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList $taskWatchArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_watcher.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_watcher.stderr.log')) -PassThru
@{watcher_pid=$taskWatcher.Id;owner_pid=$taskProcess.Id;started_at=$taskStart.ToString('o');logs_outside_candidate=$true;independent_of_heartbeat=$true}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $taskObserve ($taskPrefix+'_watcher_launch.json')) -Encoding UTF8
$taskMonitorPath=Join-Path $taskRoot 'state/biohub_t4_monitor.json'
$taskMonitor=Get-Content -LiteralPath $taskMonitorPath -Raw|ConvertFrom-Json
$taskMonitor.remote_refs|Add-Member -Force NoteProperty $Candidate ([pscustomobject]@{ref=$taskAction.ref;version=1;phase='private_train_audit';status_path=$taskRelative+'/status.json';launch_path=$taskRelative+'/launch.json';pid=$taskProcess.Id;estimated_completion=$taskEta.ToString('o');first_check=$taskEta.AddMinutes(-10).ToString('o');research_only_never_submit=$true})
$taskMonitor.updated=$taskStart.ToString('o')
$taskMonitor|ConvertTo-Json -Depth 20|Set-Content -LiteralPath $taskMonitorPath -Encoding UTF8
$taskLaunch|ConvertTo-Json -Depth 8
