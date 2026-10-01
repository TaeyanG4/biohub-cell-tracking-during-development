param([ValidateSet('c069','c070')][string]$Candidate='c069')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskRelative="experiments/candidates/${Candidate}_fixed_appearance/remote"
$taskOut=Join-Path $taskRoot $taskRelative
if((Test-Path (Join-Path $taskOut 'launch.json')) -or (Test-Path (Join-Path $taskOut 'status.json'))){throw 'Existing watcher; inspect without relaunch'}
$taskAction=Get-Content (Join-Path $taskOut 'push_action.json') -Raw|ConvertFrom-Json
if($taskAction.status -ne 'pushed' -or $taskAction.version -ne 1){throw 'No exact successful v1 push'}
$taskObserve=Join-Path $taskRoot 'state/background_notifications'
$taskPrefix=$Candidate+'_clean_T4'
$taskProcess=Start-Process -FilePath 'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe' -ArgumentList @('-X','utf8','-u','src/c069_remote.py','watch','--candidate',$Candidate) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_owner.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_owner.stderr.log')) -PassThru
$taskStart=[DateTimeOffset]::Now
$taskEta=$taskStart.AddMinutes(30)
$taskLaunch=[ordered]@{candidate=$Candidate;phase='clean_production_T4';pid=$taskProcess.Id;process_start_time=$taskProcess.StartTime.ToString('o');started_at=$taskStart.ToString('o');estimated_completion=$taskEta.ToString('o');first_check=$taskEta.AddMinutes(-10).ToString('o');check_interval_minutes=20;estimated_minutes=30;estimate_basis='Actual parent15min T4 plus appearance/setup/queue margin15min';ref=$taskAction.ref;version=1;no_competition_submission=$true}
$taskLaunch|ConvertTo-Json -Depth 8|Set-Content (Join-Path $taskOut 'launch.json') -Encoding UTF8
$taskWatcher=Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','tools/notify_background_completion.ps1','-Mode','Watch','-Label',($Candidate.ToUpper()+'-clean-T4'),'-StatusPath',($taskRelative+'/status.json'),'-LaunchPath',($taskRelative+'/launch.json'),'-ReceiptPath',('state/background_notifications/'+$taskPrefix+'_completion.json')) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_watcher.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_watcher.stderr.log')) -PassThru
@{watcher_pid=$taskWatcher.Id;owner_pid=$taskProcess.Id;started_at=$taskStart.ToString('o');logs_outside_candidate=$true}|ConvertTo-Json|Set-Content (Join-Path $taskObserve ($taskPrefix+'_watcher_launch.json')) -Encoding UTF8
$taskLaunch|ConvertTo-Json -Depth 8
