# Run a command INSIDE the project venv.
# Usage: & .claudein\with-venv.ps1 python -V
#        $env:VENV_DIR='E:\drift\.venv-other'; & .claudein\with-venv.ps1 pip list
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$venv = if ($env:VENV_DIR) { $env:VENV_DIR } else { Join-Path $repo '.venv-targets' }

if (Test-Path (Join-Path $venv 'Scripts')) { $bindir = Join-Path $venv 'Scripts' }
elseif (Test-Path (Join-Path $venv 'bin')) { $bindir = Join-Path $venv 'bin' }
else { Write-Error "with-venv: no venv at $venv"; exit 1 }

$clean = (($env:PATH -split ';') | Where-Object {
    $_ -and ($_ -notmatch 'venv[^\/]*[\/](Scripts|bin)[\/]?$')
}) -join ';'

$env:VIRTUAL_ENV = $venv
$env:PATH = "$bindir;$clean"
if (Test-Path Env:PYTHONHOME) { Remove-Item Env:PYTHONHOME }

if ($args.Count -eq 0) { "VIRTUAL_ENV=$env:VIRTUAL_ENV"; exit 0 }
& $args[0] @($args | Select-Object -Skip 1)
exit $LASTEXITCODE
