param([Parameter(Mandatory=$true)][ValidateSet('c065','c066','c067','c068')][string]$Candidate)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskFolders=@{c065='c065_pooled_transformer/deploy';c066='c066_expanded_ordinary/deploy';c067='c067_single_source_44b6';c068='c068_single_source_6bba'}
$taskRelative='experiments/candidates/'+$taskFolders[$Candidate]+'/remote'
$taskOut=Join-Path $taskRoot $taskRelative
if((Test-Path -LiteralPath (Join-Path $taskOut 'launch.json')) -or (Test-Path -LiteralPath (Join-Path $taskOut 'status.json'))){throw 'Existing production watcher; no duplicate'}
$taskAction=Get-Content -LiteralPath (Join-Path $taskOut 'push_action.json') -Raw|ConvertFrom-Json
if($taskAction.status -ne 'pushed' -or -not $taskAction.clean_production){throw 'Clean production v1 push not confirmed'}
$taskObserve=Join-Path $taskRoot 'state/background_notifications'
$taskPython='C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe'
$taskPrefix=$Candidate+'_clean_T4'
$taskArgs=@('-X','utf8','-u','src/final_four_remote.py','watch','--candidate',$Candidate)
$taskProcess=Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_owner.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_owner.stderr.log')) -PassThru
$taskStart=[DateTimeOffset]::Now
$taskEta=([DateTimeOffset]::Parse($taskAction.ended)).AddMinutes(25)
$taskLaunch=[ordered]@{candidate=$Candidate;phase='clean_production_T4';pid=$taskProcess.Id;process_start_time=$taskProcess.StartTime.ToString('o');started_at=$taskStart.ToString('o');remote_ref=$taskAction.ref;version=1;estimated_completion=$taskEta.ToString('o');first_check=$taskEta.AddMinutes(-10).ToString('o');check_interval_minutes=20;estimated_minutes=25;estimate_basis='Prior exact T4 visible4 inference12-15min plus10min setup/queue margin; queuing may extend';source='src/final_four_remote.py';source_sha256=(Get-FileHash (Join-Path $taskRoot 'src/final_four_remote.py') -Algorithm SHA256).Hash.ToLower();no_competition_submission=$true}
$taskLaunch|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $taskOut 'launch.json') -Encoding UTF8
$taskWatchArgs=@('-NoProfile','-ExecutionPolicy','Bypass','-File','tools/notify_background_completion.ps1','-Mode','Watch','-Label',($Candidate.ToUpper()+'-T4'),'-StatusPath',($taskRelative+'/status.json'),'-LaunchPath',($taskRelative+'/launch.json'),'-ReceiptPath',('state/background_notifications/'+$taskPrefix+'_completion.json'))
$taskWatcher=Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList $taskWatchArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_watcher.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_watcher.stderr.log')) -PassThru
@{watcher_pid=$taskWatcher.Id;owner_pid=$taskProcess.Id;started_at=$taskStart.ToString('o');logs_outside_candidate=$true;independent_of_heartbeat=$true}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $taskObserve ($taskPrefix+'_watcher_launch.json')) -Encoding UTF8
$taskMonitorPath=Join-Path $taskRoot 'state/biohub_t4_monitor.json'
$taskMonitor=Get-Content -LiteralPath $taskMonitorPath -Raw|ConvertFrom-Json
$taskMonitor.remote_refs|Add-Member -Force NoteProperty ($Candidate+'_clean') ([pscustomobject]@{ref=$taskAction.ref;version=1;phase='clean_production_T4';status_path=$taskRelative+'/status.json';launch_path=$taskRelative+'/launch.json';pid=$taskProcess.Id;estimated_completion=$taskEta.ToString('o');first_check=$taskEta.AddMinutes(-10).ToString('o');no_competition_submission=$true})
$taskMonitor.updated=$taskStart.ToString('o')
$taskMonitor|ConvertTo-Json -Depth 20|Set-Content -LiteralPath $taskMonitorPath -Encoding UTF8
$taskLaunch|ConvertTo-Json -Depth 8
