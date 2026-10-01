param([ValidateSet('c069','c070')][string]$Candidate='c069')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskRelative="experiments/candidates/${Candidate}_fixed_appearance/study"
$taskOut=Join-Path $taskRoot $taskRelative
if((Test-Path -LiteralPath (Join-Path $taskOut 'launch.json')) -or (Test-Path -LiteralPath (Join-Path $taskOut 'status.json'))){throw 'Existing phase; inspect without relaunch'}
$taskOther=if($Candidate -eq 'c069'){'c070'}else{'c069'}
$taskOtherStatus=Join-Path $taskRoot "experiments/candidates/${taskOther}_fixed_appearance/study/status.json"
if(Test-Path $taskOtherStatus){if((Get-Content $taskOtherStatus -Raw|ConvertFrom-Json).status -eq 'running'){throw 'Other appearance phase owns GPU'}}
if([DateTimeOffset]::Now.AddMinutes(50+15+30+15) -gt [DateTimeOffset]::Parse('2026-09-30T00:00:00+09:00')){throw 'Full remaining path does not fit midnight'}
$taskObserve=Join-Path $taskRoot 'state/background_notifications'
$taskPrefix=$Candidate+'_appearance'
$taskProcess=Start-Process -FilePath 'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe' -ArgumentList @('-X','utf8','-u','src/c069_appearance_replacement.py','run','--candidate',$Candidate) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_owner.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_owner.stderr.log')) -PassThru
$taskStart=[DateTimeOffset]::Now
$taskEta=$taskStart.AddMinutes(50)
$taskLaunch=[ordered]@{candidate=$Candidate;phase='fixed_appearance_actual97';pid=$taskProcess.Id;process_start_time=$taskProcess.StartTime.ToString('o');started_at=$taskStart.ToString('o');estimated_completion=$taskEta.ToString('o');first_check=$taskEta.AddMinutes(-10).ToString('o');check_interval_minutes=20;estimated_minutes=50;estimate_basis='Historical97 off+on59min; one on arm plus2off controls and actual97 expected50min';source='src/c069_appearance_replacement.py';source_sha256=(Get-FileHash (Join-Path $taskRoot 'src/c069_appearance_replacement.py')).Hash.ToLower();no_primary_inference=$true;deepcenter_can_use_cuda=$true;no_kaggle_writes=$true}
$taskLaunch|ConvertTo-Json -Depth 8|Set-Content (Join-Path $taskOut 'launch.json') -Encoding UTF8
$taskWatcher=Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','tools/notify_background_completion.ps1','-Mode','Watch','-Label',($Candidate.ToUpper()+'-appearance97'),'-StatusPath',($taskRelative+'/status.json'),'-LaunchPath',($taskRelative+'/launch.json'),'-ReceiptPath',('state/background_notifications/'+$taskPrefix+'_completion.json')) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskObserve ($taskPrefix+'_watcher.stdout.log')) -RedirectStandardError (Join-Path $taskObserve ($taskPrefix+'_watcher.stderr.log')) -PassThru
@{watcher_pid=$taskWatcher.Id;owner_pid=$taskProcess.Id;started_at=$taskStart.ToString('o');logs_outside_candidate=$true}|ConvertTo-Json|Set-Content (Join-Path $taskObserve ($taskPrefix+'_watcher_launch.json')) -Encoding UTF8
$taskLaunch|ConvertTo-Json -Depth 8
