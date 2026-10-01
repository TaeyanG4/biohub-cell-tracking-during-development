param(
    [ValidateSet('Watch','Test','Check')][string]$Mode = 'Watch',
    [string]$StatusPath,
    [string]$LaunchPath,
    [string]$Label = 'Biohub experiment',
    [string]$ReceiptPath
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskAppId = 'OpenAI.Codex_2p2nqsd0c76g0!App'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$taskNotifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($taskAppId)
if ($taskNotifier.Setting.ToString() -ne 'Enabled') { throw ('Windows toast disabled: ' + $taskNotifier.Setting) }
if ($Mode -eq 'Check') { Write-Output ('Windows toast setting: ' + $taskNotifier.Setting); exit 0 }

function Send-CompletionToast([string]$Title, [string]$Body, [string]$Tag) {
    $taskXml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $taskSafeTitle = [System.Security.SecurityElement]::Escape($Title)
    $taskSafeBody = [System.Security.SecurityElement]::Escape($Body)
    $taskXml.LoadXml('<toast><visual><binding template="ToastGeneric"><text>' + $taskSafeTitle + '</text><text>' + $taskSafeBody + '</text></binding></visual><audio src="ms-winsoundevent:Notification.Default"/></toast>')
    $taskToast = [Windows.UI.Notifications.ToastNotification]::new($taskXml)
    $taskToast.Tag = $Tag
    $taskToast.Group = 'BiohubJobs'
    $taskToast.ExpirationTime = [DateTimeOffset]::Now.AddHours(12)
    $taskNotifier.Show($taskToast)
}

if ($Mode -eq 'Test') {
    Send-CompletionToast 'Biohub 알림 설정 완료' '백그라운드 작업이 완료되거나 실패하면 Windows 알림과 소리로 알려드립니다.' 'setup-test'
    Write-Output 'Native Windows test notification submitted.'
    exit 0
}

if (-not $StatusPath -or -not $LaunchPath -or -not $ReceiptPath) { throw 'Watch requires StatusPath, LaunchPath and ReceiptPath.' }
$taskStatusFull = [IO.Path]::GetFullPath((Join-Path $taskRoot $StatusPath))
$taskLaunchFull = [IO.Path]::GetFullPath((Join-Path $taskRoot $LaunchPath))
$taskReceiptFull = [IO.Path]::GetFullPath((Join-Path $taskRoot $ReceiptPath))
foreach ($taskPath in @($taskStatusFull,$taskLaunchFull,$taskReceiptFull)) {
    if (-not $taskPath.StartsWith($taskRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Path outside this workspace.' }
}
$taskLaunch = Get-Content -LiteralPath $taskLaunchFull -Raw -Encoding UTF8 | ConvertFrom-Json
$taskOwnerPid = [int]$taskLaunch.pid
$taskOwnerStart = [DateTimeOffset]::Parse($taskLaunch.process_start_time)
$taskDeadline = [DateTimeOffset]::Now.AddHours(24)
$taskOutcome = $null
$taskDetail = $null
while ([DateTimeOffset]::Now -lt $taskDeadline) {
    $taskState = $null
    if (Test-Path -LiteralPath $taskStatusFull) {
        try { $taskState = Get-Content -LiteralPath $taskStatusFull -Raw -Encoding UTF8 | ConvertFrom-Json } catch { Start-Sleep -Seconds 2; continue }
    }
    if ($taskState -and $taskState.status -in @('complete_review_required','complete','completed','success','failed')) {
        $taskOutcome = [string]$taskState.status
        $taskDetail = if ($taskOutcome -eq 'failed') { [string]$taskState.error } else { '계산 및 검증 명령이 끝났습니다. 결과 검토가 이어집니다.' }
        break
    }
    $taskOwner = Get-Process -Id $taskOwnerPid -ErrorAction SilentlyContinue
    if (-not $taskOwner -or [Math]::Abs((([DateTimeOffset]$taskOwner.StartTime)-$taskOwnerStart).TotalSeconds) -gt 1) {
        $taskOutcome = 'unexpected_exit'
        $taskDetail = '정상 완료 기록 없이 프로세스가 종료되었습니다. 실행 로그 확인이 필요합니다.'
        break
    }
    Start-Sleep -Seconds 5
}
if (-not $taskOutcome) { $taskOutcome='watch_timeout'; $taskDetail='24시간 동안 완료되지 않았습니다. 실행 상태 확인이 필요합니다.' }
$taskEvent = $taskOwnerPid.ToString() + '|' + $taskOwnerStart.ToString('o') + '|' + $taskOutcome
if (Test-Path -LiteralPath $taskReceiptFull) {
    $taskOld = Get-Content -LiteralPath $taskReceiptFull -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskOld.event -eq $taskEvent -and $taskOld.toast_submitted) { Write-Output 'Already notified.'; exit 0 }
}
$taskFailed = $taskOutcome -in @('failed','unexpected_exit','watch_timeout')
$taskTitle = $Label + $(if ($taskFailed) { ' 실패 / 확인 필요' } else { ' 완료' })
if ($taskDetail.Length -gt 220) { $taskDetail=$taskDetail.Substring(0,220) }
Send-CompletionToast $taskTitle $taskDetail ('job-' + $taskOwnerPid)
$taskReceipt = [ordered]@{event=$taskEvent;label=$Label;status=$taskOutcome;toast_submitted=$true;timestamp=[DateTimeOffset]::Now.ToString('o');pid=$taskOwnerPid;process_start_time=$taskOwnerStart.ToString('o');notifier_setting=$taskNotifier.Setting.ToString();title=$taskTitle;body=$taskDetail;source='local Windows notification API; independent of Codex automation'}
$taskReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taskReceiptFull -Encoding UTF8
$taskReceipt | ConvertTo-Json -Depth 5
