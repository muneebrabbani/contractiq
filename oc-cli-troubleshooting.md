# Troubleshooting: `oc` command not recognized in PowerShell

## Issue

Running the OpenShift CLI failed with:

```
PS E:\Personal\atomcamp\final_project\code> oc version --client
oc : The term 'oc' is not recognized as the name of a cmdlet, function, script file, or operable program.
Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ oc version --client
+ ~~
    + CategoryInfo          : ObjectNotFound: (oc:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
```

## Root cause

- `oc.exe` (~106 MB) was present directly in the project working directory (`E:\Personal\atomcamp\final_project\code`), confirmed via `git status` showing it as an untracked file (`?? oc.exe`).
- PowerShell does **not** execute commands from the current directory by default, even if the file is right there. This is a security default (unlike `cmd.exe` on old Windows or `bash` with `.` in `$PATH`).
- Since the folder wasn't on `PATH` either, PowerShell couldn't resolve the bare `oc` command.

## Diagnosis steps

1. Confirmed the file actually existed in the directory:
   ```powershell
   ls -la oc.exe
   ```
   Output confirmed the binary was present and executable.

2. Ran it with an explicit relative path to bypass PATH resolution:
   ```powershell
   .\oc.exe version --client
   ```
   This succeeded:
   ```
   Client Version: 4.21.0-202603242057.p2.g56a56a3.assembly.stream.el9-56a56a3
   Kustomize Version: v5.7.1
   ```

## Fix applied

Added the project directory to the **User** `PATH` environment variable (persistent, not just session-scoped):

```powershell
$dir = "E:\Personal\atomcamp\final_project\code"
$current = [Environment]::GetEnvironmentVariable("Path", "User")
$new = if ($current) { "$current;$dir" } else { $dir }
[Environment]::SetEnvironmentVariable("Path", $new, "User")
```

## Caveats / follow-up

- **New terminals only**: The updated `PATH` only takes effect in newly opened shells. The terminal session where this was run still needs `.\oc.exe` until it's restarted (or `refreshenv` is used, if installed).
- **Fragile location**: `oc.exe` currently lives inside a git project working directory. If this folder is ever deleted, moved, or cleaned (`git clean -fd`, etc.), the `oc` command will break system-wide.
- **Recommended cleanup**: Move `oc.exe` to a dedicated tools folder outside the repo, e.g. `%USERPROFILE%\bin`, and point `PATH` there instead. Also consider adding `oc.exe` to `.gitignore` in the meantime so it doesn't get accidentally committed to the repo.

## Verification

To verify the PATH change took effect in a fresh terminal:
```powershell
oc version --client
```
Should now resolve without needing `.\` prefix.
