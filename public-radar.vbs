Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)

' Ensure custom protocol handler kaggle-radar: is registered in user registry
On Error Resume Next
shell.RegWrite "HKCU\Software\Classes\kaggle-radar\", "URL:Kaggle Radar Protocol", "REG_SZ"
shell.RegWrite "HKCU\Software\Classes\kaggle-radar\URL Protocol", "", "REG_SZ"
shell.RegWrite "HKCU\Software\Classes\kaggle-radar\shell\open\command\", "wscript.exe """ & WScript.ScriptFullName & """ ""%1""", "REG_SZ"
On Error GoTo 0

openBrowser = True
For Each arg In WScript.Arguments
  argLower = LCase(arg)
  If argLower = "--no-browser" Or InStr(argLower, "kaggle-radar:") > 0 Then
    openBrowser = False
  End If
Next

pythonw = "C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\pythonw.exe"
If Not fso.FileExists(pythonw) Then
  pythonw = "pythonw.exe"
End If

launcher = root & "\tools\notebook_radar\notebook_radar.py"
shell.CurrentDirectory = root
' Run pythonw completely hidden (0 = SW_HIDE, False = async without waiting)
shell.Run """" & pythonw & """ """ & launcher & """ serve --port 8792", 0, False

If openBrowser Then
  WScript.Sleep 1500
  shell.Run "http://127.0.0.1:8792", 1, False
End If
