# Run a command OUTSIDE any venv (system Python).
# Usage: & .claudein
o-venv.ps1 python -V
$env:PATH = (($env:PATH -split ';') | Where-Object {
    $_ -and ($_ -notmatch 'venv[^\/]*[\/](Scripts|bin)[\/]?$')
}) -join ';'
if (Test-Path Env:VIRTUAL_ENV) { Remove-Item Env:VIRTUAL_ENV }
if (Test-Path Env:PYTHONHOME)  { Remove-Item Env:PYTHONHOME }

if ($args.Count -eq 0) { "VIRTUAL_ENV=<unset>"; exit 0 }
& $args[0] @($args | Select-Object -Skip 1)
exit $LASTEXITCODE
